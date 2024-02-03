# FACT-AI
Repository containing code for the FACT-AI project

# Running on experiment Snellius (temporary)

First, load the required modules:

```
module load 2022
module load IPython/8.5.0-GCCcore-11.3.0 # Python/3.10.4 instead works as well
module load CUDA/11.7.0 # 12.1.0 should also work
```

create a virtual environment and install the required packages:

```
virtualenv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

There is a slight inconvinience with torch_scatter module. So please run the following command line to add torch_scatter module to the environment.

```
pip install torch_scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```


The command lines to reproduce our results on Slurm cluster are as follows:

Claim 1 -> Accuracy, DP, and EO of Graphair with our best parameters on NBA, Pokec-n, and Pokec-z can be found in results/claim_1/claim_1_output.txt

```
sbatch claim_1.job
```

Claim 2 -> Accuracy, DP, and EO of Graphair, Graphair w/o FM, and Graphair w/o EP with the same parameters from claim 1 on NBA, Pokec-n, and Pokec-z can be found in results/claim_2/claim_2_output.txt

```
sbatch claim_2.job
```

Claim 3 -> Plots of Homophily-Density and Spearman Correlations of Graphair on NBA, Pokec-n, Pokec-z can be found in results/claim_3/

```
sbatch claim_3.job
```

Link Prediction -> Results of Graphair on link prediction task on Citeseer, Cora, PubMed can be found in results/link_prediction

```
sbatch link_prediction.job
```


