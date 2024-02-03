import time
import json
from itertools import product

import torch
from models.fairgraph.method import run
from models.fairgraph.dataset import POKEC, NBA, Citeseer, Cora, PubMed
from utils import set_seed, parse_arguments, create_writer

start = time.process_time()
# Parse command line arguments
args, adjusted_args = parse_arguments()

datasets = {
    'pokec':POKEC,
    'nba':NBA,
    'citeseer':Citeseer,
    'cora':Cora,
    'pubmed':PubMed
}

MODEL = 'LPGraphair' if args.lp else 'Graphair'

# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('running inference with the following args:\n')
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

hyperparameters = {
    'alpha': args.alphas,
    'beta': args.betas,
    'gamma': args.gammas,
    'lam': args.lams,
    'lr': args.lr,
    'model_lr': args.model_lr,
    'hidden': args.hidden
}

writer = create_writer(args.dataset, MODEL, args.epochs, hyperparameters)

if args.state == 'fm':
    disable_fm, disable_ep = True, False
elif args.state == 'ep':
    disable_fm, disable_ep = False, True
else:
    disable_fm, disable_ep = False, False

for idx, params in enumerate(product(*hyperparameters.values())):
    set_seed(20)
    alpha, beta, gamma, lam, lr, model_lr, hidden = params

    if 'pokec' in args.dataset:
        dataset = datasets['pokec'](dataset_sample=args.dataset)
    else:
        dataset = datasets[args.dataset]()
    run_fairgraph = run()
    print(f'running with params: alpha = {alpha}, beta = {beta}, gamma = {gamma}, '
            f'lamda = {lam}, lr = {lr}, model lr = {model_lr}, hidden_size = {hidden}')
    results = run_fairgraph.run(device, dataset=dataset, model=MODEL,
                                epochs=args.epochs, test_epochs=args.test_epochs,
                                batch_size=1000, lr=lr, model_lr=model_lr, weight_decay=args.wd,
                                hidden=hidden, alpha=alpha, beta=beta, gamma=gamma, lam=lam, disable_ep=disable_ep, disable_fm=disable_fm)

    hparams = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'lam': lam,
            'lr': lr,
            'model_lr': model_lr,
            'hidden': hidden,
            'dataset':args.dataset,
            'model':MODEL,
            'test_epochs':args.test_epochs
            }

    metrics = {metric: value[0] for metric, value in results.items() if metric not in ['homophily', 'spearman']}
    results = {metric: value for metric, value in results.items() if metric not in ['homophily', 'spearman']}
    writer.add_hparams(hparams, metrics)
    
    with open(args.result_path, 'a', encoding='utf-8') as f:
        title = f"Result for dataset = {args.dataset.upper()}" if args.state == 'normal' else f"Result for dataset = {args.dataset.upper()}, state = w/o {args.state.upper()}"
        f.write(title +'\n')
        f.write('-' * 50 + '\n')
        f.flush()

        f.write(f"Accuracy ([acc, std]): {results['accuracy']}\n")
        f.write(f"DP ([dp, std]): {results['dp']}\n")
        f.write(f"EO ([eo, std]): {results['eo']}\n")

        f.write('-' * 50 + '\n')
        f.write('\n'*2)
        f.flush()
        end = time.process_time()
        f.write(f'The code needed {(end - start) / 60} minutes to run\n')
        f.write(f'The maximum amount of memory used was {torch.cuda.max_memory_allocated() / (1024 ** 3)} GB.\n')
        f.flush()

writer.close()
