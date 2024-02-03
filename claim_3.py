###############################################
# The file should work. Waiting for the node...
# TODO:
# - update the plots with correct range and smooth
###############################################

from models.fairgraph.dataset import NBA, POKEC
from models.fairgraph.method import run
import torch
import matplotlib.pyplot as plt
from models.fairgraph.utils.utils import weighted_homophily, scipysp_to_pytorchsp, spearman_correlation
import seaborn as sns
import numpy as np
import time

start = time.process_time()

# nba, smooth fair 13, original 13
# pokec-z, smooth fair 20, original 35
# pokec-n, smooth fair 15, original 37

model = 'Graphair'
datasets = {'nba': {'dataset': NBA(),
                  'epochs': 500,
                  'test_epochs': 500, 
                  'batch_size': 1000,
                  'lr': 0.001,
                  'model_lr': 0.0001,
                  'hidden': 128,
                  'alpha': 1.0,
                  'beta': 0.1,
                  'gamma': 0.1,
                  'lam': 1.0,
                  'yrange': [0.325, 0.525],
                  'smooth': [13, 13]
                  },
        'pokec_n': {'dataset': POKEC(dataset_sample='pokec_n'),
                  'epochs': 500,
                  'test_epochs': 500, 
                  'batch_size': 1000,
                  'lr': 1e-3,
                  'model_lr': 1e-4,
                  'hidden': 128,
                  'alpha': 0.1,
                  'beta': 1.0,
                  'gamma': 0.1,
                  'lam': 10.0,
                  'yrange': [0.1, 0.8],
                  'smooth': [15, 37]
                  },
        'pokec_z': {'dataset': POKEC(dataset_sample='pokec_z'),
                  'epochs': 500,
                  'test_epochs': 500, 
                  'batch_size': 1000,
                  'lr': 1e-3,
                  'model_lr': 1e-4,
                  'hidden': 128,
                  'alpha': 10.0,
                  'beta': 10.0,
                  'gamma': 0.1,
                  'lam': 10.0,
                  'yrange': [0.1, 0.8],
                  'smooth': [20, 35]
                  }
}

filename = f'results/claim_3/claim_3_output.txt'

# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_fairgraph = run()


with open(filename, 'w', encoding='utf-8') as f:
    for data_name, params in datasets.items():
        f.write(f"Result for claim 3: dataset = {data_name.upper()}" +'\n')
        f.write('-' * 50 + '\n')
        f.flush()

        adj = scipysp_to_pytorchsp(params['dataset'].adj).to_dense()

        result = run_fairgraph.run(device, dataset=params['dataset'], model=model,
                                        epochs=params['epochs'], test_epochs=params['test_epochs'],
                                        batch_size=params['batch_size'], lr=params['lr'], model_lr=params['model_lr'], weight_decay=1e-5,
                                        hidden=params['hidden'], alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'], lam=params['lam'])

        # Homophily 
        homophily_values_fair = result["homophily"]
        homophily_values_original = weighted_homophily(adj, params['dataset'].sens)

        sns.kdeplot(homophily_values_fair, bw_adjust=params['smooth'][0], label='Fair view', color='orange')
        sns.kdeplot(homophily_values_original, bw_adjust=params['smooth'][1], label='Original', color='blue')
        plt.xlim(0, 1)
        plt.ylim(params['yrange'][0], params['yrange'][1])
        plt.axvline(x=np.mean(homophily_values_fair), color='orange', linestyle='--')
        plt.axvline(x=np.mean(homophily_values_original), color='blue', linestyle='--')
        plt.text(np.mean(homophily_values_fair), plt.ylim()[0], f'{np.mean(homophily_values_fair):.2f}', color='orange', ha='left',va='bottom')
        plt.text(np.mean(homophily_values_original), plt.ylim()[0], f'{np.mean(homophily_values_original):.2f}', color='blue', ha='left', va='bottom')
        plt.title(f'{data_name.upper()}')
        plt.xlabel('Homophily Value', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.legend()
        plt.savefig(f'results/claim_3/homophily_{data_name}.png')
        plt.clf()

        # Spearman
        spearman_correlations_fair = result["spearman"][0]
        spearman_correlations_original = spearman_correlation(params['dataset'].features, params['dataset'].sens)

        indexed_list = list(enumerate(spearman_correlations_original))
        sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
        top_10_indices = [index for index, _ in sorted_list[:10]]
        spearman_correlations_fair = [spearman_correlations_fair[i] for i in top_10_indices]
        spearman_correlations_original = [spearman_correlations_original[i] for i in top_10_indices]

        indices = range(10)
        plt.bar(indices, spearman_correlations_original, width=0.4, label='Original')
        plt.bar([i + 0.4 for i in indices], spearman_correlations_fair, width=0.4, label='Fair view')
        plt.xlabel('Feature index', fontsize=15)
        plt.ylabel('Spearman correlation', fontsize=15)
        plt.title(data_name.upper())
        plt.legend()
        plt.xticks([i + 0.2 for i in indices], indices)
        plt.savefig(f'results/claim_3/spearman_{data_name}.png')
        plt.clf()
    
    end = time.process_time()
    f.write(f'The code needed {(end - start) / 60} minutes to run\n')
    f.write(f'The maximum amount of memory used was {torch.cuda.max_memory_allocated() / (1024 ** 3)} GB.')
    f.flush()
