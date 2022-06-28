import pandas as pd
import subprocess
import sys
from string import Template

# read input file
df = pd.read_csv(sys.argv[1])

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
    runner_script = template.substitute(MODEL_ID=row.model_id,
                                        MODEL_TYPE=row.model_type,
                                        MODEL_AREA=row.model_area)

    # write temporary script file
    try:
        temp_script = open("./batch_analysis/.temp_script.sh", "w")
        n = temp_script.write(runner_script)
        temp_script.close()
    except:
        print("Cannot create temp script.")

    # assign execute right to temp script
    subprocess.run(["chmod", "+x", "./batch_analysis/.temp_script.sh"])

    # run temp script
    output = subprocess.run(["sbatch", "./batch_analysis/.temp_script.sh"], stdout=subprocess.PIPE)

    # remove temp script
    subprocess.run(["rm", "./batch_analysis/.temp_script.sh"])