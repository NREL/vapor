# --- Imports ---
import logging
import random
import json
import os
import concurrent.futures as cf
import itertools
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import vapor.config as config

log = logging.getLogger("vapor")

nrel_color_dict = ['#0077C8', #darkblue
                '#00A9E0', #lightblue
                '#658D1B', #darkgreen
                '#84BD00', #lightgreen
                '#FFC72C', #yellow
                '#DE7C00', #orange
                '#5B6770', #darkgray
                '#C1C6C8'] #lightgray

tech_dict = {'pv':nrel_color_dict[4],
            'wind':nrel_color_dict[1],
            'batt':nrel_color_dict[3]}

batt_size_dict = {0:'o', 25:'P', 100:'^'}

scen_label_dict = {
    'StdScen20_LowRECost':'Low Wholesale Cost',
    'StdScen20_MidCase':'Baseline Cost',
    'StdScen20_HighRECost':'High Wholesale Cost'}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ VISUALIZATION OF OUTPUT ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Visualizer():
    def __init__(self, results, aggregate_region,
                legend=True,
                 last_year=config.LAST_YEAR,
                 region_label_pct=0.4):
        
        assert isinstance(legend, bool)
        assert isinstance(results, pd.DataFrame)
        
        self.legend = legend
        self.last_year = last_year
        self.region_label_pct = region_label_pct #put labels on geographys larger than this percentile size. 0 for no labels.
    
        # --- Merge on geometry to data ---
        geo = gpd.read_file(os.path.join('data','geography','ReEDS_Resource_Regions.shp'))
        geo = geo[[aggregate_region, 'geometry']]
        geo.rename({aggregate_region:'region'}, inplace=True, axis='columns')
        geo = geo.dissolve(by='region')

        self.gdf = geo.merge(results, on='region', how='inner')

        self.cbarloc = [0.2, 0.2, 0.2, 0.05] #location for legend color bar
        self.colorscheme = 'Reds' #matplotlib colorscheme
        self.units = None
        
    def _update_crs(self, crs=2163):
        """Update the CRS to US National Atlas Standard (non-meractor)."""
        self.gdf= self.gdf.to_crs(f'epsg:{crs}')
    
    def _update_units(self, column, round_at=1):
        """Update units for the selected column."""
        
        unit_df = self.gdf.copy()

        MW_size = unit_df['system_capacity'] / 1000

        if column in ['cambium_co2_rate_marg', 'cambium_co2_rate_avg', 'lifetime_co2_rate_marginal']:
            self.units = 'Mil Tons'
            unit_df[column] = unit_df[column] / 1000000 #convert from kg/W to MT/MW
            self.suffix = 'Savings'
        
        elif column in ['lcoe_real','lcoe_nom', 'lppa_nom', 'lppa_real','ppa']:
            self.units = '$/MWh'
            unit_df[column] = unit_df[column] * 10 #convert from Cents/kWh to $/MWh
            self.colorscheme = self.colorscheme + '_r' #reverse colormap
            self.suffix = 'Price'
            
        elif column in ['cambium_enduse_energy_value','cambium_busbar_energy_value','cambium_grid_value',
                        'cambium_capacity_value','cambium_as_value','cambium_portfolio_revenue']:
            self.units = '$/MW/yr'
            unit_df[column] = unit_df[column] / MW_size #TODO: check units
            self.suffix = 'Savings'

        elif column in ['project_return_aftertax_irr']:
            self.units = '%'
            self.suffix = ''

        elif column in ['project_return_aftertax_npv', 'lifetime_cambium_grid_value']:
            unit_df[column] = unit_df[column] / 1000000
            self.units = '$ Mil'
            self.suffix = ''

        elif column in ['marginal_cost_mwh']:
            self.units = '$/MWh'
            self.suffix = ''
        
        else:
            self.units = ''
            self.suffix = ''
            
        # --- Round ---
        unit_df[column] = unit_df[column].round(round_at)
        return unit_df
        
        
    def merged_choropleth(self, column, scenario, batt_size=0, ascending=False, storage=True, reverse_cmap=False, title=None, *kwargs):

        # --- Update crs ---
        self._update_crs()
        
        # --- Update units ---
        plot_df = self._update_units(column)

        # --- subset scenario ---
        assert scenario in plot_df['scenario'].unique()
        plot_df = plot_df.loc[(plot_df['scenario'] == scenario) & (plot_df['batt_size'] == batt_size)]

        # --- pick best tech for each geo ---
        plot_df.sort_values(column, ascending=ascending, inplace=True)
        plot_df.drop_duplicates(subset=['region'], keep='first', inplace=True)
        
        # --- Clean up column string ---
        clean_column = column.replace('_',' ').title()
        clean_column = f"{clean_column} {self.suffix}"
        
        fig, ax = plt.subplots(dpi=200)

        for tech in plot_df['tech'].unique():

            tech_df = plot_df.loc[plot_df['tech'] == tech]
            
            if tech == 'pv':
                cmap = 'Oranges'
            elif tech == 'wind':
                cmap = 'Blues'

            if reverse_cmap:
                cmap = cmap + '_r'
            
            tech_df.plot(column, edgecolor='k', cmap=cmap, alpha=0.8, linewidth=0.2, ax=ax)
        
            # if self.legend:
            #     vmax = max(tech_df[column])
            #     vmed = round(tech_df[column].mean(),0)
            #     vmin = min(tech_df[column])
        
            #     fig = ax.get_figure()
            #     cax = fig.add_axes(self.cbarloc) #set size and location of cbar
                
            #     # --- Create array of values for color bar ---
            #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            #     sm._A = []
            #     cbar = fig.colorbar(sm, cax=cax, alpha = 0.8, orientation='horizontal', ticks = [vmin,vmed,vmax])
            #     cbar.ax.set_xticklabels([f"{int(vmin):,}", f"{int(vmed):,}\n{self.units}", f"{int(vmax):,}"])
            
            if self.region_label_pct > 0:
                
                tech_df['area_pct'] = tech_df.area.rank(pct=True)
                
                for _, row in tech_df.iterrows():
                    if row['area_pct'] > self.region_label_pct:
                        ax.annotate(s=f"{round(row[column], 1):,}",
                                    xy=row['geometry'].centroid.coords[0],
                                    horizontalalignment='center',
                                    size=7)
        
        ax.axis('off')
        
        if title == None:
            ax.set_title(f"{clean_column} ({self.units}) through {self.last_year}")
        else:
            ax.set_title(title)
    
    def choropleth(self, column, scenario, tech, batt_size=0, ascending=False, storage=True, reverse_cmap=False, title=None, *kwargs):

        # --- Update crs ---
        self._update_crs()
        
        # --- Update units ---
        plot_df = self._update_units(column)

        # --- subset scenario ---
        assert scenario in plot_df['scenario'].unique()
        plot_df = plot_df.loc[(plot_df['scenario'] == scenario) & (plot_df['batt_size'] == batt_size) & (plot_df['tech'] == tech)]

        # --- pick best tech for each geo ---
        plot_df.sort_values(column, ascending=ascending, inplace=True)
        plot_df.drop_duplicates(subset=['region'], keep='first', inplace=True)
        
        # --- Clean up column string ---
        clean_column = column.replace('_',' ').title()
        clean_column = f"{clean_column} {self.suffix}"
        
        fig, ax = plt.subplots(dpi=200)
            
        if tech == 'pv':
            cmap = 'Oranges'
        elif tech == 'wind':
            cmap = 'Blues'

        if reverse_cmap:
            cmap = cmap + '_r'
            
        plot_df.plot(column, edgecolor='k', cmap=cmap, alpha=0.8, linewidth=0.2, ax=ax, scheme='Percentiles')
        
        if self.region_label_pct > 0:
            
            plot_df['area_pct'] = plot_df.area.rank(pct=True)
            
            for _, row in plot_df.iterrows():
                if row['area_pct'] > self.region_label_pct:
                    ax.annotate(text=f"{round(row[column], 1):,}",
                                xy=row['geometry'].centroid.coords[0],
                                horizontalalignment='center',
                                size=7)
        
        ax.axis('off')
        
        if title != None:
            ax.set_title(title)

    def triple_choropleth(self, column, tech, batt_size=0, ascending=False, storage=True, reverse_cmap=False, title=None, *kwargs):

        # --- Update crs ---
        self._update_crs()
        
        # --- Update units ---
        plot_df = self._update_units(column)

        # --- subset scenario ---
        plot_df = plot_df.loc[(plot_df['batt_size'] == batt_size)]
        cutoffs = plot_df[column].quantile(list(np.arange(0,1,0.2))).to_list()
        classification_kwds = dict(bins=cutoffs)
        plot_df = plot_df.loc[(plot_df['tech'] == tech)]

        # --- Clean up column string ---
        clean_column = column.replace('_', ' ').title()
        clean_column = f"{clean_column} {self.suffix}"

        if tech == 'pv':
            cmap = 'Oranges'
        elif tech == 'wind':
            cmap = 'Blues'
        if reverse_cmap:
            cmap = cmap + '_r'

        fig, axs = plt.subplots(figsize=(8,4), dpi=400, ncols=3)

        for i, s in enumerate(['StdScen20_LowRECost', 'StdScen20_MidCase', 'StdScen20_HighRECost']):
            scenario_df = plot_df.loc[plot_df['scenario'] == s]

            # --- pick best tech for each geo ---
            scenario_df.sort_values(column, ascending=ascending, inplace=True)
            scenario_df.drop_duplicates(subset=['region'], keep='first', inplace=True)
            scenario_df.plot(column, edgecolor='k', cmap=cmap, alpha=0.8, linewidth=0.2, ax=axs[i],
                            scheme='UserDefined', classification_kwds=classification_kwds)
        
            if self.region_label_pct > 0:
                
                scenario_df['area_pct'] = scenario_df.area.rank(pct=True)
                
                for _, row in scenario_df.iterrows():
                    if row['area_pct'] > self.region_label_pct:
                        axs[i].annotate(text=f"{round(row[column], 1):,}",
                                    xy=row['geometry'].centroid.coords[0],
                                    horizontalalignment='center',
                                    size=7)
        
            axs[i].axis('off')
            axs[i].set_title(scen_label_dict[s], fontsize=10)
            plt.tight_layout()


    def supply_curve(self, column, scenario, label='Marginal Price\n($/MWh)', batt_sizes=None, ascending=True, legend=True):

        # --- Update units ---
        plot_df = self._update_units(column)

        # --- subset scenario ---
        no_batt = plot_df.loc[(plot_df['scenario'] == scenario) & (plot_df['batt_size'] == 0)]

        # --- drop duplicate techs for same state ---
        no_batt.sort_values(column, ascending=ascending, inplace=True)
        no_batt.drop_duplicates(subset=['region', 'scenario'], inplace=True, keep='first')

        # --- mock up widths and bar positions and color ---
        width = [i for i in no_batt['system_capacity']]

        relative_positions = []
        relative_position = 0
        previous_width = 0
        for w in width:
            relative_position = float(relative_position + (previous_width/2) + (w/2))
            previous_width = w
            relative_positions.append(relative_position)
            
        colors = no_batt['tech'].map(tech_dict)
            
        fig, ax = plt.subplots(figsize=(8,2), dpi=400)
        ax.bar(relative_positions, no_batt[column],
                width=width, linewidth=0.0,
                color=colors)
                
        # --- mock up label positions ---
        odd = -1
        for i, l in enumerate(no_batt['region']):
            x = relative_positions[i]
            y_max = list(no_batt[column])[i]
            y =  y_max / 2 + (odd * y_max * 0.075)
            ax.annotate(l, xy=(x,y), ha='center',va='bottom',
                    fontsize=6)
            odd *= -1

        # --- plot battery bars ---
        if batt_sizes != None:
            for batt_size in batt_sizes:
                batt = plot_df.loc[(plot_df['scenario'] == scenario) & (plot_df['batt_size'] == batt_size)]
                batt.sort_values(column, ascending=ascending, inplace=True)
                batt.drop_duplicates(subset=['region', 'scenario'], inplace=True, keep='first')

                ax.bar(relative_positions, batt[column],
                    width=width, linewidth=0.0,
                    color=nrel_color_dict[3], alpha=0.5, zorder=0)

        # --- clean up ---
        plt.title(f'{scen_label_dict[scenario]}', fontsize=10)
        plt.subplots_adjust(hspace=0.5)
        plt.ylabel(label)
        plt.xlabel('')#Cumulative RE Capacity')
        # plt.ylim(-5, 70)
        plt.tight_layout()

        if legend:
            label_dict = {'pv':'Solar PV', 'wind':'Wind', 'batt':'w/ 4 hr Battery'}
            custom_patches = [mpatches.Patch(color=v, label=label_dict[k]) for k,v in tech_dict.items()]
            plt.legend(handles=custom_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        sns.despine(fig)

    
    def scatter_facet(self, xcol, ycol, legend=False):
        
        # --- Update units ---
        plot_df = self.gdf.copy()
        
        sns.set_style()
        fig, axs = plt.subplots(figsize=(10,6), nrows=2, ncols=3, sharey=True, dpi=400)


        for i_scenario, s in enumerate(['StdScen20_LowRECost', 'StdScen20_MidCase', 'StdScen20_HighRECost']):
            for i_tech, tech in enumerate(['pv','wind']):
                for b in set(plot_df['batt_size']):
                    scenario_df = plot_df.loc[(plot_df['scenario'] == s) & (plot_df['batt_size'] == b) & (plot_df['tech'] == tech)]

                    colors = scenario_df['tech'].map(tech_dict)
                    label = f"{tech.capitalize()} - {int(b)} MW Batt"
                    axs[i_tech][i_scenario].scatter(
                            x=scenario_df[xcol],
                            y=scenario_df[ycol],
                            s=80,
                #                s=scenario_df[zcol] * 2,
                            c=colors, marker=batt_size_dict[b],
                            alpha=0.3, edgecolor="k", linewidth=0.5,
                            label=label)

                    poly = np.poly1d(np.polyfit(scenario_df[xcol], scenario_df[ycol], 2))
                    x_min = scenario_df[xcol].min() * 0.9
                    x_max = scenario_df[xcol].max() * 1.1
                    x_range = np.linspace(x_min, x_max, 50)
                    axs[i_tech][i_scenario].plot(
                                x_range, poly(x_range),
                                c=tech_dict[tech], linewidth=3, alpha=0.8)

                if i_scenario == 0:
                    axs[i_tech][i_scenario].set_ylabel('Lifetime Cumulative \n Grid Value')

                if i_tech == 1:
                    axs[i_tech][i_scenario].set_xlabel('Marginal Cost ($/MWh)')
                
                if i_tech ==0:
                    axs[i_tech][i_scenario].set_title(scen_label_dict[s])
                    axs[i_tech][i_scenario].get_xaxis().set_visible(False)

                axs[i_tech][i_scenario].set_xlim(0, 100)
                axs[i_tech][i_scenario].set_ylim(0.4*1e8, 1.8*1e8)
        
        if legend:
            solar_handles, solar_labels = axs[i_tech-1][i_scenario].get_legend_handles_labels()
            wind_handles, wind_labels = axs[i_tech][i_scenario].get_legend_handles_labels()
            fig.legend(solar_handles+wind_handles, solar_labels+wind_labels, bbox_to_anchor=(1.2, 1))

        sns.despine()
        plt.tight_layout()

