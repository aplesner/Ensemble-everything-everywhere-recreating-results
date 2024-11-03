#DEFINE SBATCH STUFF

#SBATCH ARRAY 1-5


cd ~/code/path/to/code

source helper_script/project_variables.sh
mkdir /scratch/$USERNAME

export HUGGING...="/scratch/..."

for model in []
apptainer exec --bind /scratch/USRENAME --bind /itet-stor/... --nv SINGULARITY_FILE python script.py $TASK_ID other stuff $model
mv results.log /itet-stor/...
