"""
This batch scheduler will queue jobs on HPC (eagle), requesting one node for each job, allowing us to run multiple scenarios, technologies and configurations in parallel.

If you are just trying to run a single scenario, or run vapor on a laptop, invoke the high-level functions available in run_cli.py

Implementation:
1. Walk through nested loops of possible variables
2. Write bash file to 'bash_files' dir
3. Submit sbatch job using bash_file
"""

import os
import subprocess
import time

# --- HPC env variables ---
HPC = 'eagle'

# Eagle:
SLURM_ACCOUNT = 'vapor'
SLURM_MIN_DURATION = '00:60:00'
SLURM_MAX_DURATION = '03:00:00'
SLURM_OUTPUT_DIR = 'slurm_out/%j.out'
CONDA_ENV = 'vp'
USER = 'skoebric'
SLURM_QOS = 'high'

# --- Define Optimization Variables ---
mode = 'constraint'
aggregate_region = 'state'
opt_vars = ['adjusted_installed_cost']
scenarios = ['StdScen19_Mid_Case']
techs = ['wind','pv']
goal_pcts = [25,50,75,95]
goal_types = ['hourly_energy']

def write_bash(opt_var, scenario, tech, goal_pct, goal_type):

    if HPC == 'eagle':
        script = (
                '#!/bin/bash\n'
                f"#SBATCH --account={SLURM_ACCOUNT}\n"
                f"#SBATCH --time-min={SLURM_MIN_DURATION}\n"
                f"#SBATCH --time={SLURM_MAX_DURATION}\n"
                f"#SBATCH --qos={SLURM_QOS}\n"
                f"#SBATCH --output={SLURM_OUTPUT_DIR}\n"
                f"#SBATCH --nodes=1\n\n"

                f"module load conda\n"
                f"source deactivate\n"
                f"source activate {CONDA_ENV}\n\n"

                f"python run_cli.py --scenario={scenario} --tech={tech} --opt_var={opt_var} --aggregate_region={aggregate_region} --mode={mode} --goal_pct={goal_pct} --goal_type={goal_type}"
        )
        
        fp = os.path.join('bash_files', f"{aggregate_region}_{opt_var}_{scenario}_{tech}_{goal_pct}_{goal_type}.sh")
        with open(fp, "w") as text_file:
            text_file.write(script)

        subprocess.call(f'sbatch {fp}', shell=True)
        print(f"Submitted {fp} to SLURM\n")
    
    elif HPC == 'hermes':
        script = f"python run_cli.py --scenario={scenario} --tech={tech} --opt_var={opt_var} --aggregate_region={aggregate_region} --mode={mode} --goal_pct={goal_pct} --goal_type={goal_type}"
        breakpoint()
        subprocess.call(script, shell=True)


for opt_var in opt_vars:
    for scenario in scenarios:
        for tech in techs:
            for goal_pct in goal_pcts:
                for goal_type in goal_types:
                    write_bash(opt_var, scenario, tech, goal_pct, goal_type)


# --- print squeue ---
if HPC == 'eagle':
    while True:
        print('\n')
        print('Run Status:')
        subprocess.call(f'squeue -u {USER}', shell=True)
        time.sleep(180)
