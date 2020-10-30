import vapor
import pandas as pd
import os
import argparse

import logging
log = logging.getLogger("vapor")

def regional(scenario, tech, aggregate_region, batt_size, batt_duration, opt_var):
    log.info(f"Working on regional run for {scenario}, {opt_var}, {tech}, {batt_size}/{batt_duration} batt")

    # --- Initialize Pipeline ---
    plumbing = vapor.RegionalPipeline(
                scenario=scenario,
                tech=tech,
                aggregate_region=aggregate_region,
                re_capacity_mw=100,
                batt_capacity_mw=batt_size,
                batt_duration=batt_duration
    )

    # --- Execute Simulations ---
    plumbing.setup()
    plumbing.run([opt_var])

    # --- Get Best Systems ---
    best = plumbing.best_systems

    # --- Save ---
    best['scenario'] = scenario
    best['batt_size'] = batt_size
    best.to_pickle(os.path.join('results',f"{aggregate_region}_best_{tech}_{scenario}_{opt_var}_batt_{batt_size}_{batt_duration}.pkl"))


def existing(scenario, tech, aggregate_region, batt_size, batt_duration, opt_var):
    log.info(f"Working on existing run for {scenario}, {tech}, {batt_size}/{batt_duration} batt")

    vapor.config.BAYES_INIT_POINTS = 0
    vapor.config.BAYES_ITER = 0

    bnef = vapor.load_bnef()

    # --- Initialize Pipeline ---
    plumbing = vapor.ExistingPipeline(
        aggregate_region=aggregate_region,
        scenario=scenario,
    )

    plumbing.setup(df=bnef)
    bnef = plumbing.fetch_resource(bnef)
    plumbing.run(df=bnef,
                opt_vars=[opt_var])

    # --- Save ---
    best = plumbing.best_systems
    best.to_pickle(os.path.join('results', 'bnef_results.pkl'))

def constrain(scenario, tech, aggregate_region, opt_var, goal_pct, goal_type):
    log.info(f"Working on constraint run for {scenario}, {tech} with {goal_pct} {goal_type}")

    dummy_annual_load_mwh = 100000

    if goal_type in ['hourly_energy', 'hourly_co2']:
        upper_bound_system_size = (dummy_annual_load_mwh / (0.1 * 8760)) * 100 #we initially probe this value, divided by [1, 2.5, 5, 10, 25, 50, 100]
        upper_bound_batt_size = upper_bound_system_size
    elif goal_type in ['annual_recs']:
        upper_bound_system_size = (dummy_annual_load_mwh / (0.1 * 8760)) * 100 #we initially probe this value, divided by [1, 2.5, 5, 10, 25, 50, 100]
        upper_bound_batt_size = 0 #assuming battery will never be economic in annual calculations

    # --- Initialize Pipeline ---
    plumbing = vapor.GoalPipeline(
        scenario=scenario,
        tech=tech,
        aggregate_region=aggregate_region,
        re_capacity_mw=[2.5, upper_bound_system_size],
        batt_capacity_mw=[0., upper_bound_batt_size],
        batt_duration=[0, 2, 4],
        annual_load_mwh=dummy_annual_load_mwh,
        goal_type=goal_type,
        goal_pct=goal_pct,
    )

    # --- Execute Simulations ---
    plumbing.setup()
    plumbing.run(opt_vars=[opt_var])

    # --- Get Best Systems ---
    best = plumbing.best_systems

    # --- Save ---
    best['scenario'] = scenario
    best.to_pickle(os.path.join('results',f"{aggregate_region}_best_{tech}_{scenario}_{opt_var}_constraint_{goal_type}_{goal_pct}.pkl"))


if __name__ == "__main__":
    # --- CLI arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='StdScen19_Mid_Case')
    parser.add_argument('--tech', type=str, default='pv')
    parser.add_argument('--aggregate_region', type=str, default='census_reg')
    parser.add_argument('--batt_size', type=float, default=100)
    parser.add_argument('--batt_duration', type=float, default=4)
    parser.add_argument('--opt_var', type=str, default='marginal_cost_mwh')
    parser.add_argument('--mode', type=str, default='regional')
    parser.add_argument('--goal_pct', type=int, default=70)
    parser.add_argument('--goal_type', type=str, default='hourly_energy')

    # --- Parse args ---
    args = parser.parse_args()
    args_dict = vars(args)
    mode = args_dict.pop('mode')

    # --- Run Vapor ---
    if mode == 'regional':
        del args_dict['goal_pct']
        del args_dict['goal_type']
        regional(**args_dict)
    if mode == 'existing':
        del args_dict['goal_pct']
        del args_dict['goal_type']
        existing(**args_dict)
    if mode == 'constraint':
        del args_dict['batt_size']
        del args_dict['batt_duration']
        constrain(**args_dict)

