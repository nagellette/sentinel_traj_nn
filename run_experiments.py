import subprocess
from string import Template
import sys
import pandas as pd

# read input summary file
df = pd.read_csv(sys.argv[1], sep="|")
df["slurm_id"] = ""

# start runner loop
for i, row in df.iterrows():
    # open template file specified as argument
    try:
        f = open(sys.argv[2], "r")
        template = f.read()
    except:
        print("File {} not available".format(sys.argv[1]))

    template = Template(template)

    # add actual values to template
    runner_script = template.substitute(JOBDEVICE=row.JOBDEVICE,
                                        JOBNAME=row.JOBNAME,
                                        MODELNAME=row.MODELNAME,
                                        MODELCONFIG=row.MODELCONFIG,
                                        MODELINPUTS=row.MODELINPUTS)

    # write temporary script file
    try:
        temp_script = open("./experiments/.temp_script.sh", "w")
        n = temp_script.write(runner_script)
        temp_script.close()
    except:
        print("Cannot create temp script.")

    # assign execute right to temp script
    subprocess.run(["chmod", "+x", "./experiments/.temp_script.sh"])

    # run temp script
    output = subprocess.run(["sbatch", "./experiments/.temp_script.sh"], stdout=subprocess.PIPE)

    # catch script output and write to dataframe
    output = output.stdout.decode("utf-8").split(" ")
    slurm_id = output[-1].replace("\n", "")
    df.loc[i, "slurm_id"] = slurm_id

    # remove temp script
    subprocess.run(["rm", "./experiments/.temp_script.sh"])

# write dataframe with slurm job IDs
df.to_csv(sys.argv[1].replace(".csv", "_out.csv"), index=False)
