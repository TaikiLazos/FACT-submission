import numpy as np
import scipy.sparse as sp
import torch
import scipy.stats
from sklearn.metrics import roc_auc_score

def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def auc(output, labels):
    output = output.squeeze()
    
    # device = output.device
    # labels = labels.to(device)
    
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    # # Sort outputs and get indices
    # sorted_indices = output.argsort(descending=True)
    # sorted_labels = labels[sorted_indices]
    
    # # Calculate the number of positive and negative examples
    # n_pos = labels.sum()
    # n_neg = len(labels) - n_pos
    
    # # Rank positive examples; ranks start from 1
    # rank = torch.arange(1, len(labels) + 1)
    # pos_ranks = rank[sorted_labels.bool()]
    
    # # Sum of ranks of positive examples
    # pos_rank_sum = pos_ranks.sum()
    
    # # Calculate AUC using the formula: (Sum(pos ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    auc = roc_auc_score(labels, output)
    return auc


def fair_metric(output,idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity * 100 ,equality * 100

def define_sens_variables_from_adj(adj_matrix, node_sens_attributes):
    """
    Define sensitive variables for link prediction from an adjacency matrix.

    :param adj_matrix: Adjacency matrix of the graph (scipy sparse matrix).
    :param node_sens_attributes: Array of sensitive attributes for each node.
    :return: te_mixed_dyadic, te_dyadic_src, te_dyadic_dst, te_sub_diatic
    """
    # Extract edge indices from the adjacency matrix
    edge_indices = np.array(sp.find(adj_matrix)[:2])

    te_dyadic_src = node_sens_attributes[edge_indices[0]]
    te_dyadic_dst = node_sens_attributes[edge_indices[1]]

    # Mixed Dyadic
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst

    # # Subgroup Dyadic
    # unique_combinations = list(combinations_with_replacement(np.unique(node_sens_attributes), 2))
    # te_sub_diatic = np.array([unique_combinations.index(tuple(sorted((src, dst))))
    #                           for src, dst in zip(te_dyadic_src, te_dyadic_dst)])
    te_sub_diatic = None
    
    return te_mixed_dyadic, te_dyadic_src, te_dyadic_dst, te_sub_diatic
    
def weighted_homophily(adj_matrix, sens):
    node_homophily = np.zeros(adj_matrix.shape[0])

    sens = sens.cpu()

    for i in range(adj_matrix.shape[0]):
        neighbors = adj_matrix[i, :]
        same_label_strength = neighbors[sens == sens[i]].sum()
        total_strength = neighbors.sum()

        if total_strength > 0:
            node_homophily[i] = same_label_strength / total_strength
        else:
            node_homophily[i] = 0

    return node_homophily

def spearman_correlation(features, sens):
    correlations = []

    sens = sens.cpu().numpy()

    for i in range(features.shape[1]):
        correlation, _ = scipy.stats.spearmanr(sens, features[:, i].cpu().numpy())
        correlations.append(correlation)

    return correlations