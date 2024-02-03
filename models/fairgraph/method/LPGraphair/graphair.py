import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from models.fairgraph.utils.utils import scipysp_to_pytorchsp,accuracy,fair_metric, auc
import random
from itertools import combinations_with_replacement

import torch_scatter
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_adj, to_torch_sparse_tensor, to_edge_index, add_remaining_self_loops

class graphair(nn.Module):
    r'''
        This class implements the Graphair model

        :param aug_model: The augmentation model g described in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ used for automated graph augmentations
        :type aug_model: :obj:`torch.nn.Module`

        :param f_encoder: The represnetation encoder f described in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ used for contrastive learning
        :type f_encoder: :obj:`torch.nn.Module`

        :param sens_model: The adversary model k described in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ used for adversarial learning
        :type sens_model: :obj:`torch.nn.Module`

        :param classifier_model: The classifier used to predict the sensitive label of nodes on the augmented graph data.
        :type classifier_model: :obj:`torch.nn.Module`

        :param lr: Learning rate for aug_model, f_encoder and sens_model. Defaults to 1e-4
        :type lr: float,optional

        :param weight_decay: Weight decay for regularization. Defaults to 1e-5
        :type weight_decay: float,optional

        :param alpha: The hyperparameter alpha used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to scale adversarial loss component. Defaults to 20.0
        :type alpha: float,optional

        :param beta: The hyperparameter beta used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to scale contrastive loss component. Defaults to 0.9
        :type beta: float,optional

        :param gamma: The hyperparameter gamma used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to scale reconstruction loss component. Defaults to 0.7
        :type gamma: float,optional

        :param lam: The hyperparameter lambda used in the `paper <https://openreview.net/forum?id=1_OGWcP1s9w>`_ to compute reconstruction loss component. Defaults to 1.0
        :type lam: float,optional

        :param dataset: The name of the dataset being used. Used only for the model's output path. Defaults to 'POKEC'
        :type dataset: str,optional

        :param batch_size: The batch size paramter used for minibatch creation. Used only for the model's output path. Defaults to None
        :type batch_size: int,optional

        :param num_hidden: The input dimension for the MLP networks used in the model. Defaults to 64
        :type num_hidden: int,optional

        :param num_proj_hidden: The output dimension for the MLP networks used in the model. Defaults to 64
        :type num_proj_hidden: int,optional

    '''
    def __init__(
        self,
        aug_model,
        f_encoder,
        sens_model,
        classifier_model,
        lr=1e-4,
        model_lr=1e-4,
        weight_decay=1e-5,
        alpha=20,
        beta=0.9,
        gamma=0.7,
        lam=1,
        dataset="POKEC",
        batch_size=None,
        num_hidden=64,
        num_proj_hidden=64,
    ):
        super(graphair, self).__init__()
        self.aug_model = aug_model
        self.f_encoder = f_encoder
        self.sens_model = sens_model
        self.classifier = classifier_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dataset = dataset
        self.lam = lam
        

        # self.criterion_sens = nn.BCEWithLogitsLoss()
        self.criterion_sens = nn.CrossEntropyLoss()
        self.criterion_cont= nn.CrossEntropyLoss()
        self.criterion_recons = nn.MSELoss()

        self.optimizer_s = torch.optim.Adam(
            self.sens_model.parameters(), lr=model_lr, weight_decay=1e-5
        )

        FG_params = [
            {"params": self.aug_model.parameters(), "lr": 1e-4},
            {"params": self.f_encoder.parameters()},
        ]
        self.optimizer = torch.optim.Adam(
            FG_params, lr=model_lr, weight_decay=weight_decay
        )

        self.optimizer_aug = torch.optim.Adam(
            self.aug_model.parameters(), lr=model_lr, weight_decay=weight_decay
        )
        self.optimizer_enc = torch.optim.Adam(
            self.f_encoder.parameters(), lr=model_lr, weight_decay=weight_decay
        )

        self.batch_size = batch_size

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.optimizer_classifier = torch.optim.Adam(
            self.classifier.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        print(lr)
    
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def info_nce_loss_2views(self, features):
        
        batch_size = int(features.shape[0] / 2)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        temperature = 0.07
        logits = logits / temperature
        return logits, labels

    # this function we should change to not return node embeddings, but link embeddings
    def forward(self, adj, x, sens):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)
    
        adj = adj_norm.cuda()


        # MODIFIED FROM HERE

        # Get node embeddings
        z = self.f_encoder(adj, x)
        
        # Create positive and negative link embeddings and labels
        pos_rows, pos_cols = adj.coalesce().indices()
        neg_rows, neg_cols = self.sample_negative_links(adj, num_neg_samples=pos_rows.shape[0])
        pos_labels = torch.ones(pos_rows.shape[0], dtype=torch.float32)
        neg_labels = torch.zeros(neg_rows.shape[0], dtype=torch.float32)
        
        rows = torch.cat([torch.tensor(pos_rows, device='cpu'), torch.tensor(neg_rows, device='cpu')], dim=0)
        cols = torch.cat([torch.tensor(pos_cols, device='cpu'), torch.tensor(neg_cols, device='cpu')], dim=0)

        
        link_embeddings = self.create_link_embeddings(z, rows, cols)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Create mixed dyadic groups
        groups_mixed = sens[rows] != sens[cols]

        # Create dyadic subgroups
        u = list(combinations_with_replacement(np.unique(sens.cpu()), r=2))
        groups_sub = []
        for i, j in zip(sens[rows], sens[cols]):
            for k, v in enumerate(u):
                if (i, j) == v or (j, i) == v:
                    groups_sub.append(k)
                    break
        groups_sub = np.asarray(groups_sub)

        return link_embeddings, labels, groups_mixed, groups_sub

    # ADDITIONAL FUNCTION
    def create_link_embeddings(self, node_embeddings, row_indices, col_indices):
        # Using a simple binary operator, e.g., element-wise addition
        link_embeddings = node_embeddings[row_indices] * node_embeddings[col_indices]

        return link_embeddings
    
    # ADDITIONAL FUNCTION
    def sample_negative_links(self, adj, num_neg_samples):
        n = adj.size(0)
        neg_rows, neg_cols = [], []
        while len(neg_rows) < num_neg_samples:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            # Check if the edge exists in the sparse adjacency matrix
            if adj._indices()[0][adj._indices()[1] == i].ne(j).all():
                neg_rows.append(i)
                neg_cols.append(j)
        return np.array(neg_rows), np.array(neg_cols)
    
    def fit_batch(self, epochs, adj, x,sens,idx_sens,warmup=None, adv_epoches=1):
        print("########################")
        print("epochs:", epochs)
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = sp.csr_matrix(adj)
        norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)

        idx_sens = idx_sens.cpu().numpy()
        sens_mask = np.zeros((x.shape[0],1))
        sens_mask[idx_sens] = 1.0
        sens_mask = torch.from_numpy(sens_mask)

        edge_index, _ = from_scipy_sparse_matrix(adj)

        miniBatchLoader = GraphSAINTRandomWalkSampler(Data(x=x, edge_index=edge_index, sens = sens, sens_mask = sens_mask), 
                                                        batch_size = 1000, 
                                                        walk_length = 3, 
                                                        sample_coverage = 500, 
                                                        num_workers = 0,
                                                        save_dir = "./checkpoint/{}".format(self.dataset))

        def normalize_adjacency(adj):
            # Calculate the degrees
            row, col = adj.indices()
            edge_weight = adj.values() if adj.values() is not None else torch.ones(row.size(0))
            degree = torch_scatter.scatter_add(edge_weight, row, dim=0, dim_size=adj.size(0))

            # Inverse square root of degree matrix
            degree_inv_sqrt = degree.pow(-0.5)
            degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

            # Normalize
            row_inv = degree_inv_sqrt[row]
            col_inv = degree_inv_sqrt[col]
            norm_edge_weight = edge_weight * row_inv * col_inv

            # Create the normalized sparse tensor
            adj_norm = torch.sparse.FloatTensor(torch.stack([row, col]), norm_edge_weight, adj.size())
            return adj_norm

        if warmup:
            for _ in range(warmup):
                for data in miniBatchLoader:
                    data = data.cuda()
                    edge_index,_ = add_remaining_self_loops(data.edge_index)
                    sub_adj = normalize_adjacency(to_torch_sparse_tensor(edge_index)).cuda()
                    sub_adj_dense = to_dense_adj(edge_index = edge_index, max_num_nodes = data.x.shape[0])[0].float()
                    adj_aug, x_aug, adj_logits = self.aug_model(sub_adj, data.x, adj_orig = sub_adj_dense)  

                    edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, sub_adj_dense)

                    feat_loss =  self.criterion_recons(x_aug, data.x)
                    recons_loss =  edge_loss + self.lam * feat_loss

                    self.optimizer_aug.zero_grad()
                    with torch.autograd.set_detect_anomaly(True):
                        recons_loss.backward(retain_graph=True)
                    self.optimizer_aug.step()

                    print(
                    'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                    'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                    )

     # or .cpu() depending on your device
        
        for epoch_counter in range(epochs):
            for data in miniBatchLoader:
                data = data.cuda()

                ### generate fair view
                edge_index,_ = add_remaining_self_loops(data.edge_index)
                sub_adj = normalize_adjacency(to_torch_sparse_tensor(edge_index)).cuda()

                sub_adj_dense = to_dense_adj(edge_index = edge_index, max_num_nodes = data.x.shape[0])[0].float()
                adj_aug, x_aug, adj_logits = self.aug_model(sub_adj, data.x, adj_orig = sub_adj_dense)  


                ### extract node representations
                h = self.projection(self.f_encoder(sub_adj, data.x))
                h_prime = self.projection(self.f_encoder(adj_aug, x_aug))

                ### update sens model
                adj_aug_nograd = adj_aug.detach()
                x_aug_nograd = x_aug.detach()

                mask = (data.sens_mask == 1.0).squeeze()

                
                if (epoch_counter == 0):
                    sens_epoches = adv_epoches * 10
                else:
                    sens_epoches = adv_epoches
                    
                if self.dataset == 'Citeseer':
                    nclasses = 6
                elif self.dataset == 'Cora':
                    nclasses = 7
                elif self.dataset == 'PubMed':
                    nclasses = 3
                else:
                    nclasses = 1

                class_counts = [sum(data.sens == i) for i in range(nclasses)]
                class_weights = [1.0 / count if count > 0 else 0 for count in class_counts]
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda()

                for _ in range(sens_epoches):

                    s_pred , _  = self.sens_model(adj_aug_nograd, x_aug_nograd)

                    senloss = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='sum')(s_pred[mask], data.sens[mask].long())

                    self.optimizer_s.zero_grad()
                    senloss.backward()
                    self.optimizer_s.step()
                
                s_pred , _  = self.sens_model(adj_aug, x_aug)
                senloss = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='sum')(s_pred[mask], data.sens[mask].long())

                ## update aug model
                logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim = 0))
                contrastive_loss = (nn.CrossEntropyLoss(reduction='none')(logits, labels) * data.node_norm.repeat(2)).sum()

                ## update encoder
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, sub_adj_dense)    

                feat_loss =  self.criterion_recons(x_aug, data.x)
                recons_loss =  edge_loss + self.lam * feat_loss
                loss = self.beta * contrastive_loss + self.gamma * recons_loss - self.alpha * senloss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Epoch: {:04d}'.format(epoch_counter+1),
            'sens loss: {:.4f}'.format(senloss.item()),
            'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
            'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
            'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
            )

        self.save_path = "./checkpoint/graphair_{}".format(self.dataset)
        torch.save(self.state_dict(),self.save_path)
        
    def fit_whole(self, epochs, adj, x,sens,idx_sens,warmup=None, adv_epoches=1):
        # print(idx_sens)
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = scipysp_to_pytorchsp(adj).to_dense()
        print("---------------------adj_orig---------------------")
        print(adj_orig)
        # print how many links are in the dataset
        print("number of links in dataset:", adj_orig.sum())
        norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)
        

        adj = adj_norm.cuda()
        
        best_contras = float("inf")

        warmup = 0
        if warmup:
            for _ in range(warmup):
                adj_aug, x_aug, adj_logits = self.aug_model(adj, x, adj_orig = adj_orig.cuda())
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig.cuda())

                feat_loss =  self.criterion_recons(x_aug, x)
                recons_loss =  edge_loss + self.beta * feat_loss

                self.optimizer_aug.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    recons_loss.backward(retain_graph=True)
                self.optimizer_aug.step()

                print(
                'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
                'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
                )
            
        for epoch_counter in range(epochs):
            ### generate fair view
            adj_aug, x_aug, adj_logits = self.aug_model(adj, x, adj_orig = adj_orig.cuda())

            ### extract node representations
            h = self.projection(self.f_encoder(adj, x))
            h_prime = self.projection(self.f_encoder(adj_aug, x_aug))

            ## update sens model
            adj_aug_nograd = adj_aug.detach()
            x_aug_nograd = x_aug.detach()
            if (epoch_counter == 0):
                sens_epoches = adv_epoches * 10
            else:
                sens_epoches = adv_epoches
                
            for _ in range(sens_epoches):
                s_pred , _  = self.sens_model(adj_aug_nograd, x_aug_nograd)
                
                
                senloss = self.criterion_sens(s_pred[idx_sens],sens[idx_sens].long())
                
                self.optimizer_s.zero_grad()
                senloss.backward()
                self.optimizer_s.step()
                
            s_pred , _  = self.sens_model(adj_aug, x_aug)
            senloss = self.criterion_sens(s_pred[idx_sens],sens[idx_sens].long())

            ## update aug model
            logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim = 0))
            contrastive_loss = self.criterion_cont(logits, labels)

            ## update encoder
            edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig.cuda())

            feat_loss =  self.criterion_recons(x_aug, x)
            recons_loss =  edge_loss + self.lam * feat_loss
            loss = self.beta * contrastive_loss + self.gamma * recons_loss - self.alpha * senloss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('Epoch: {:04d}'.format(epoch_counter+1),
            'sens loss: {:.4f}'.format(senloss.item()),
            'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
            'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
            'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
            )

            
        self.save_path = "./checkpoint/graphair_{}_alpha{}_beta{}_gamma{}_lambda{}".format(self.dataset, self.alpha, self.beta, self.gamma, self.lam)
        torch.save(self.state_dict(),self.save_path)
    
    # modified, we dont use the input labels now,as these are returned and specific to the link embeddings
    def test(self, adj, features, labels, epochs, idx_train, idx_val, idx_test, sens):
        features = features.cuda() if torch.cuda.is_available() else features
        h, labels, groups_mixed, groups_sub = self.forward(adj, features, sens)

        # Shuffle the embeddings and labels
        indices = torch.randperm(h.size(0))
        h = h[indices]
        labels = labels[indices]
        groups_mixed = torch.tensor(groups_mixed[indices])
        groups_sub = torch.tensor(groups_sub[indices])
        h = h.detach()

        # Move indices and labels to the correct device
        device = h.device
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        labels = labels.to(device)

        acc_list = []
        roc_list = []
        dp_mixed_list = []
        eo_mixed_list = []
        dp_sub_list = []
        eo_sub_list = []

        for i in range(5):
            torch.manual_seed(i *10)
            np.random.seed(i *10)
            
            # train classifier
            self.classifier.reset_parameters()
                
            best_acc = best_roc = best_dp_mixed = best_eo_mixed = best_dp_sub = best_eo_sub = 0.0
            best_test_acc = best_test_roc = best_test_dp_mixed = best_test_eo_mixed = best_test_dp_sub = best_test_eo_sub = 0.0
                       
            for epoch in range(epochs):
                self.classifier.train()
                self.optimizer_classifier.zero_grad()
                output = self.classifier(h)
                loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
                
                loss_train.backward()
                self.optimizer_classifier.step()

                # Evaluate on validation and test sets
                self.classifier.eval()
                output = self.classifier(h)
                
                
                acc_val = accuracy(output[idx_val], labels[idx_val])
                # roc_val = auc(output[idx_val], labels[idx_val])
                acc_test = accuracy(output[idx_test], labels[idx_test])
                roc_test = auc(output[idx_test], labels[idx_test])
                

                # # Compute and print fairness metrics
                parity_val_mixed, equality_val_mixed = fair_metric(output, idx_val, labels, groups_mixed)
                parity_test_mixed, equality_test_mixed = fair_metric(output, idx_test, labels, groups_mixed)
                parity_val_sub, equality_val_sub = fair_metric(output, idx_val, labels, groups_sub)
                parity_test_sub, equality_test_sub = fair_metric(output, idx_test, labels, groups_sub)
                if epoch % 10 == 0:
                    print(
                        "Epoch [{}] Test set results:".format(epoch),
                        "acc_test= {:.4f}".format(acc_test.item()),
                        "acc_val: {:.4f}".format(acc_val.item()),
                        "dp_val: {:.4f}".format(parity_val_mixed),
                        "dp_test: {:.4f}".format(parity_test_mixed),
                        "eo_val: {:.4f}".format(equality_val_mixed),
                        "eo_test: {:.4f}".format(equality_test_mixed),
                        "dp_val: {:.4f}".format(parity_val_sub),
                        "dp_test: {:.4f}".format(parity_test_sub),
                        "eo_val: {:.4f}".format(equality_val_sub),
                        "eo_test: {:.4f}".format(equality_test_sub),
                    )
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test_acc = acc_test
                    best_test_roc = roc_test
                    best_test_dp_mixed = parity_test_mixed
                    best_test_eo_mixed = equality_test_mixed
                    best_test_dp_sub = parity_test_sub
                    best_test_eo_sub = equality_test_sub
                    
            acc_list.append(best_test_acc.detach().cpu())
            roc_list.append(best_test_roc)
            dp_mixed_list.append(best_test_dp_mixed)
            eo_mixed_list.append(best_test_eo_mixed)
            dp_sub_list.append(best_test_dp_sub)
            eo_sub_list.append(best_test_eo_sub)
        
        # Print average results
        print("Avg results:",
            "acc: {:.4f} std: {:.4f}".format(np.mean(acc_list), np.std(acc_list)),
            "AUC: {:.4f} std: {:.4f}".format(np.mean(roc_list), np.std(roc_list)),
            "dp-mixed: {:.4f} std: {:.4f}".format(np.mean(dp_mixed_list), np.std(dp_mixed_list)),
            "dp-sub: {:.4f} std: {:.4f}".format(np.mean(dp_sub_list), np.std(dp_sub_list)),
            "eo-mixed: {:.4f} std: {:.4f}".format(np.mean(eo_mixed_list), np.std(eo_mixed_list)),
            "eo-sub: {:.4f} std: {:.4f}".format(np.mean(eo_sub_list), np.std(eo_sub_list)))

        return {'accuracy': [np.mean(acc_list), np.std(acc_list)],
                'AUC': [np.mean(roc_list), np.std(roc_list)],
                'dp-mixed': [np.mean(dp_mixed_list), np.std(dp_mixed_list)],
                'eo-mixed': [np.mean(eo_mixed_list), np.std(eo_mixed_list)],
                'dp-sub': [np.mean(dp_sub_list), np.std(dp_sub_list)],
                'eo-sub': [np.mean(eo_sub_list), np.std(eo_sub_list)]}