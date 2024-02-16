# Reproducibility Study Of Learning Fair Graph Representations Via Automated Data Augmentations
Repository containing code of "Reproducibility Study Of Learning Fair Graph Representations Via Automated Data Augmentations" for TMLR submission. The reproduced paper is written by Hongyi Ling, Zhimeng Jiang, Youzhi Luo, Shuiwang Ji, Na Zou.

# Training and evaluating the model

create a virtual environment and install the required packages:

```
virtualenv my_env
source my_env/bin/activate
pip install -r requirements.txt
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

# Running model with own arguments

To test the model using your own hyperparameters, you can run the file in terminal using the following command:

```
python -m test --flag1 --flag2 ... --flag_n
```

Use the following flags to modify the training/evaluation loop:

- For the following flags, you can provide multiple values in order to rerun the model with each value, useful for grid search:

    - `--alphas`: Alpha value for training (Float).
    - `--betas`: Beta value for training (Float).
    - `--gammas`: Gamma value for training (Float).
    - `--lams`: Lambda value for training (Float).
    - `--epochs`: Number of epochs for training (Int).
    - `--model_lr`: Learning rate for the model (Float).
    - `--lr`: Learning rate for the classifier (Float).
    - `--hidden`: Hidden size for the classifier (Int).

- Provided as follows:

```
--flag *val1*  ... *val_n*
```

- For the following flags, only one value can be supplied:

    - `--test_epochs`: Number of epochs for testing (Int).
    - `--dataset`: Name of the dataset to use (String). Available choices: "pokec_z", "pokec_n", "nba", "citeseer", "cora", "pubmed".
    - `--wd`: Weight decay, (Float).
    - `--lp`: Use link prediction Graphair variant. (Flag)
    - `--result_path`: Path to save the results (String).
    - `--state`: Specify the state of the model to reproduce for claim 2. (String) By default, normal. 'ep' disables Edge Perturbation and 'fm' disables Feature Masking.
