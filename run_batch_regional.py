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
SLURM_MIN_DURATION = '04:00:00'
SLURM_MAX_DURATION = '06:00:00'
SLURM_OUTPUT_DIR = 'slurm_out/%j.out'
CONDA_ENV = 'vp_oldsam'#'vp'
USER = 'tbowen'#'skoebric'
SLURM_QOS = 'normal'

# --- Define Optimization Variables ---
mode = 'regional'
aggregate_region = 'pca'
opt_vars = ['lifetime_cambium_co2_rate_lrmer'] #['lifetime_cambium_co2_rate_avg'] #['project_return_aftertax_npv']
scenarios = ['StdScen20_HighRECost', 'StdScen20_LowRECost', 'StdScen20_MidCase']
techs = ['pv', 'wind']
batt_sizes = [0, 25, 100]
batt_durations = [4]

def write_bash(opt_var, scenario, tech, batt_size, batt_duration):

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

                f"python run_cli.py --scenario={scenario} --tech={tech} --batt_size={batt_size} --batt_duration={batt_duration} --opt_var={opt_var} --aggregate_region={aggregate_region} --mode={mode}"
        )
        
        fp = os.path.join('bash_files', f"{aggregate_region}_{opt_var}_{scenario}_{tech}_{batt_size}_{batt_duration}.sh")
        with open(fp, "w") as text_file:
            text_file.write(script)

        subprocess.call(f'sbatch {fp}', shell=True)
        print(f"Submitted {fp} to SLURM\n")

    elif HPC == 'hermes':
        script = f"python run_cli.py --scenario={scenario} --tech={tech} --batt_size={batt_size} --batt_duration={batt_duration} --opt_var={opt_var} --aggregate_region={aggregate_region} --mode={mode}"
        subprocess.call(script, shell=True)

for opt_var in opt_vars:
    for scenario in scenarios:
        for tech in techs:
            for batt_size in batt_sizes:
                for batt_duration in batt_durations:
                    write_bash(opt_var, scenario, tech, batt_size, batt_duration)


# --- print squeue ---
if HPC == 'eagle':
    while True:
        print('\n')
        print('Run Status:')
        subprocess.call(f'squeue -u {USER}', shell=True)
        time.sleep(180)
