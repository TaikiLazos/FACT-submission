from .Graphair import graphair,aug_module,GCN,GCN_Body,Classifier

from .LPGraphair import (
    graphair as lp_graphair,
    aug_module as lp_aug_module,
    GCN as lp_GCN,
    GCN_Body as lp_GCN_Body,
    Classifier as lp_Classifier
)
import time

class run():
    r"""
    This class instantiates Graphair model and implements method to train and evaluate.
    """

    def __init__(self):
        pass

    def run(self,device,dataset,model='Graphair',epochs=10_000,test_epochs=1_000,batch_size=1_000,
            lr=1e-4, model_lr=1e-4, hidden=128, weight_decay=1e-5, alpha = 20, beta = 0.9, gamma = 0.7, lam = 1):
        r""" This method runs training and evaluation for a fairgraph model on the given dataset.
        Check :obj:`examples.fairgraph.Graphair.run_graphair_nba.py` for examples on how to run the Graphair model.

        
        :param device: Device for computation.
        :type device: :obj:`torch.device`

        :param model: Defaults to `Graphair`. (Note that at this moment, only `Graphair` is supported)
        :type model: str, optional
        
        :param dataset: The dataset to train on. Should be one of :obj:`dig.fairgraph.dataset.fairgraph_dataset.POKEC` or :obj:`dig.fairgraph.dataset.fairgraph_dataset.NBA`.
        :type dataset: :obj:`object`
        
        :param epochs: Number of epochs to train on. Defaults to 10_000.
        :type epochs: int, optional

        :param test_epochs: Number of epochs to train the classifier while running evaluation. Defaults to 1_000.
        :type test_epochs: int,optional

        :param batch_size: Number of samples in each minibatch in the training. Defaults to 1_000.
        :type batch_size: int,optional

        :param lr: Learning rate. Defaults to 1e-4.
        :type lr: float,optional

        :param weight_decay: Weight decay factor for regularization. Defaults to 1e-5.
        :type weight_decay: float, optional

        :raise:
            :obj:`Exception` when model is not Graphair. At this moment, only Graphair is supported.
        """

        # Train script
        dataset_name = dataset.name
        features = dataset.features
        sens = dataset.sens
        adj = dataset.adj
        idx_sens = dataset.idx_sens_train

        # generate model
        if model=='Graphair':
            aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
            f_encoder = GCN_Body(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, dropout = 0.1, nlayer = 2).to(device)
            sens_model = GCN(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, nclass = 1).to(device)
            classifier_model = Classifier(input_dim=64,hidden_dim=hidden)
            model = graphair(aug_model=aug_model,f_encoder=f_encoder,sens_model=sens_model,classifier_model=classifier_model, lr=lr, model_lr = model_lr, weight_decay=weight_decay,
                             batch_size=batch_size,dataset=dataset_name,
                             alpha = alpha, beta = beta, gamma = gamma, lam = lam).to(device)
        elif model == 'LPGraphair':
            aug_model = lp_aug_module(features, n_hidden=64, temperature=1).to(device)
            f_encoder = lp_GCN_Body(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, dropout = 0.1, nlayer = 2).to(device)
            sens_model = lp_GCN(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, nclass = 1).to(device)
            classifier_model = lp_Classifier(input_dim=64,hidden_dim=hidden)
            model = lp_graphair(aug_model=aug_model,f_encoder=f_encoder,sens_model=sens_model,classifier_model=classifier_model, lr=lr,weight_decay=weight_decay,
                             batch_size=batch_size,dataset=dataset_name,
                             alpha = alpha, beta = beta, gamma = gamma, lam = lam).to(device)
        
        st_time = time.time()
        if dataset.batch:
            model.fit_batch(epochs=epochs,adj=adj, x=features,sens=sens,idx_sens = idx_sens, warmup=0, adv_epoches=1)
        else:
            model.fit_whole(epochs=epochs,adj=adj, x=features,sens=sens,idx_sens = idx_sens, warmup=50, adv_epoches=1)
        print("Training time: ", time.time() - st_time)
        test_results = model.test(adj=adj,features=features,labels=dataset.labels,epochs=test_epochs,idx_train=dataset.idx_train,idx_val=dataset.idx_val,idx_test=dataset.idx_test,sens=sens)
        for metric, value in test_results.items():
            if metric not in ['homophily', 'spearman']:
                print(f'{metric} = {value} ')
        time.sleep(1)
        # call test
        return test_results