#!/bin/bash



# run_experiment.sh <domain> <instance_num> <seeds> <evals> <iterations> <noise_type> <noise_std> <output_dir>

domain=$1
instance=$2
seeds=$3
evals=$4
iters=$5
noisetype=$6
noisestd=$7
output=$8

runai submit -i registry.bgu.ac.il/hpc/jupyter-notebook:latest -e HOME=/gpfs0/bgu-ataitler/users/ataitler --name "$test" -g 0.5 --cpu-limit 4 -- "cd ~/code/StochasticBPPB && source activate ~/env/StochasticBPPB && python -m StochasticBPPB.run --domain ${domain} --instance ${i} --seeds ${seeds} --eval ${evals} --iterations ${iters} --noisetype ${noisetype} --noisestd ${noisestd} -e"


# Run the Python script $count times
# for ((i=1; i<=5; i++))
# do
#  runai submit -i registry.bgu.ac.il/hpc/jupyter-notebook:latest -e HOME=/gpfs0/bgu-ataitler/users/ataitler --name "$test" -g 0.5 --cpu-limit 4 -- "cd ~/code/StochasticBPPB && source activate ~/env/StochasticBPPB && python run.py --domain ${domain} --instance ${i} --seeds ${seeds} --eval ${evals} --iterations ${iters} --noisetype ${noisetype} --noisestd ${noisestd} --output ${output} -e"
# done
