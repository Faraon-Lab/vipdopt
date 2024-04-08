#!/bin/bash
cat > $1 << EOL
#!/bin/bash
#SBATCH -A Faraon_Computing
#SBATCH --time=10:00:00   # walltime

#SBATCH --nodes=$2   # number of nodes
#SBATCH --ntasks-per-node=8  # number of processor cores (i.e. tasks)
#SBATCH --mem=16G

#SBATCH -J "lumerical sim"   # job name

#SBATCH --qos=normal

## Load relevant modules
source /home/${USER}/.bashrc
source /central/groups/Faraon_Computing/nia/miniconda3/etc/profile.d/conda.sh
conda activate vipdopt-dev

xvfb-run --server-args="-screen 0 1280x1024x24" python vipdopt optimize ${@:3}
EOL

