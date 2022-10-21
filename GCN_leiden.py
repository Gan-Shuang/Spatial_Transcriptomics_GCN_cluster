import os
import os.path as osp
import time
import anndata
import random
import warnings
import bz2
import pickle
import harmonypy
from anndata import AnnData
import scanpy as sc
import scipy
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
import torchvision
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.loader import ClusterLoader, ClusterData
import matplotlib.pyplot as plt
from typing import Optional
import sklearn
from sklearn.utils import shuffle
from datetime import datetime



warnings.filterwarnings("ignore")
def prepare_adata(dfs,norm_and_log=True,z_score=True,batch_correction=False):
    adata_list=[]
    for df in dfs:
        cell_list=df["Label"].astype("category")
        row=cell_list.cat.codes.to_numpy()
        gene_list=df["GeneName"].astype("category")
        col=gene_list.cat.codes.to_numpy()
        obs = pd.DataFrame(index=(map(str, cell_list.cat.categories)))
        var = pd.DataFrame(index=(map(str, gene_list.cat.categories)))
        pos=df.groupby("Label").mean()[["x","y"]]
        UMICount=df["UMICount"].to_numpy()
        adata_x=scipy.sparse.csr_matrix((UMICount,(row,col)), shape=(len(obs), len(var)))
        adata = AnnData(adata_x, obs=obs, var=var)
        adata.obsm['spatial'] = pos.to_numpy()
        adata_list.append(adata)
    adata = AnnData.concatenate(*adata_list,join="inner",index_unique='-')
    print("finish concat adata")
    print(datetime.fromtimestamp(int(time.time())))
    ################################################################################ normalization
    if norm_and_log:
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000.0)
        sc.pp.log1p(adata)
    if z_score:
        adata.X = adata.X.toarray()
        adata.X = (adata.X - adata.X.mean(0)) / adata.X.std(0)
    if not z_score:
        adata.X = adata.X.toarray()
    print("finish normalization")
    print(datetime.fromtimestamp(int(time.time())))
    ################################################################################ PCA & batch correction
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
#     sc.tl.pca(adata,n_comps=500) ##### pca_dim=500
    ##################### 与PCA结果有区别？
    gene_tensor = torch.Tensor(adata.X)
    u, s, v = torch.pca_lowrank(gene_tensor,q=500)
    gene_tensor = torch.matmul(gene_tensor,v)
    adata.obsm["X_pca"] = gene_tensor.numpy()
    print("finish PCA")
    print(datetime.fromtimestamp(int(time.time())))
    if batch_correction:
        sc.external.pp.harmony_integrate(adata,key="batch")
    print("finish harmony")
    print(datetime.fromtimestamp(int(time.time())))
    return(adata)



def prepare_train_loader(adata,edge_weight=True):
    data = torch_geometric.data.Data(x=torch.Tensor(adata.obsm["X_pca"]),pos=torch.Tensor(adata.obsm["spatial"]))
    data = torch_geometric.transforms.KNNGraph(k=30,loop=False)(data) ##### k_graph=30
    ################################################################################
    if edge_weight:
        data = torch_geometric.transforms.Distance()(data)
        data.edge_weight = 1 - data.edge_attr[:,0]
    else:
        data.edge_weight = torch.ones(data.edge_index.size(1))
    data.idx = torch.arange(adata.shape[0])
    ################################################################################
    cluster_data = ClusterData(data, num_parts=128) ##### num_parts=128
    train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True) ##### batch_size=32
    return(data,train_loader)



random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
class GraphEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(GraphEncoder, self).__init__()
        self.gc_feat = GCNConv(input_dims, hidden_dims)
        self.gc_mean = GCNConv(hidden_dims, output_dims)
        self.gc_var = GCNConv(hidden_dims, output_dims)
    def forward(self, x, edge_index, edge_weight):
        x = self.gc_feat(x, edge_index, edge_weight).relu()
        mean = self.gc_mean(x, edge_index, edge_weight)
        var = self.gc_var(x, edge_index, edge_weight)
        return(mean, var)

def full_block(input_dims, output_dims, drop_rate=0.2):
    return nn.Sequential(
        nn.Linear(input_dims, output_dims),
        nn.BatchNorm1d(output_dims, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=drop_rate)
    )

class SpatialModel(nn.Module):
    def __init__(self, input_dims, gae_dims, dae_dims):
        super(SpatialModel, self).__init__()
        self.input_dims = input_dims
        self.gae_dims = gae_dims
        self.dae_dims = dae_dims
        self.feat_dims = self.dae_dims[1] + self.gae_dims[1]
        self.encoder = nn.Sequential(
            full_block(self.input_dims, self.dae_dims[0]),
            full_block(self.dae_dims[0], self.dae_dims[1])
        )
        self.decoder = full_block(self.feat_dims, self.input_dims)
        self.vgae = VGAE(GraphEncoder(self.dae_dims[1], self.gae_dims[0], self.gae_dims[1]))
    def forward(self, x, edge_index, edge_weight):
        feat_x = self.encoder(x)
        feat_g = self.vgae.encode(feat_x, edge_index, edge_weight)
        feat = torch.concat([feat_x, feat_g], 1)
        x_dec = self.decoder(feat)
        dae_loss = F.mse_loss(x_dec, x)
        gae_loss = self.recon_loss(feat, edge_weight, edge_index) + 1 / len(x) * self.vgae.kl_loss()
        return(feat, dae_loss, gae_loss)
    def recon_loss(self, z, edge_weight, pos_edge_index, neg_edge_index=None):
        pos_dec = self.vgae.decoder(z, pos_edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(pos_dec, edge_weight)
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_dec = self.vgae.decoder(z, neg_edge_index)
        neg_loss = -F.logsigmoid(-neg_dec).mean()
        return(pos_loss + neg_loss)



class Trainer:
    def __init__(self, input_dims):
        self.input_dims = input_dims
        self.device = torch.device('cpu')

        gae_dims = [32, 8]
        dae_dims = [100, 24]
        self.model = SpatialModel(input_dims=self.input_dims,
                                  gae_dims=gae_dims,
                                  dae_dims=dae_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=1.0)
        self.scaler = torch.cuda.amp.GradScaler()
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    def save_checkpoint(self, path):
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, path)
    def train(self, train_loader, epochs=200, w_dae=1.0, w_gae=1.0):
        self.model.train()
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = 0
            for batch, data in enumerate(train_loader, start=1):
                data = data.to(self.device, non_blocking=True)
                inputs = data.x
                edge_index = data.edge_index
                edge_weight = data.edge_weight
                with torch.cuda.amp.autocast():
                    feat, dae_loss, gae_loss = self.model(inputs, edge_index, edge_weight)
                    loss = w_dae * dae_loss + w_gae * gae_loss
                train_loss += loss.item()
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                train_loss = train_loss / len(train_loader)
                process_time = time.time() - start_time
                print("[ Epoch %d\t Batch %d ] Loss: %.5f, Time: %.2f s" % (epoch, batch, train_loss, process_time))

    def inference(self, test_loader, cell_nums):
        self.model.eval()
        output = np.zeros((cell_nums, self.model.feat_dims))
        for data in test_loader:
            data = data.to(self.device)
            idx = data.idx.detach().cpu().numpy()
            feat, _, _ = self.model(data.x, data.edge_index, data.edge_weight)
            output[idx] = feat.detach().cpu().numpy()
        return(output)
def cluster_block(feat, adata, indices, save_path, n_neighbors=30, resolution=0.5):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('clustering ......')
    st = time.time()
    adata_feat = anndata.AnnData(feat[indices], obs=adata.obs,obsm=adata.obsm)
    adata_feat.obsm["spatial"] = adata.obsm["spatial"][indices]
    adata_feat.obsm["X_input"] = adata.obsm["X_pca"][indices]
    sc.pp.neighbors(adata_feat, n_neighbors=n_neighbors)
    sc.tl.umap(adata_feat)
    sc.tl.leiden(adata_feat, resolution=resolution)
    clusters = adata_feat.obs["leiden"].tolist()
    results = pd.DataFrame({"id": adata[indices].obs.index.tolist(),
                            "umap_x":adata_feat.obsm["X_umap"][:, 0],
                            "umap_y":adata_feat.obsm["X_umap"][:, 1], 
                            "label": clusters})
    print("cluster_results: ",results.shape)
    results.to_csv(osp.join(save_path,"cluster_batch.csv"),index=False)
    print("cluster results has been saved in path: ",save_path)
    cost_time_cluster = time.time()-st
    print("clustering finished, cost time(s): ",cost_time_cluster)
    return(cost_time_cluster,results)

epochs=100
data_dir = "./data"
save_path = './result'
################################################################################
st1=time.time()
file_names = ['batch1.data', 'batch2.data', 'batch3.data', 'batch4.data']
dfs = []
for filename in file_names:
    dfs.append(pd.read_csv(os.path.join(data_dir, filename),sep="\t"))
print("Load data cost time : {}".format(time.time()-st1))
################################################################################
st2=time.time()
adata=prepare_adata(dfs)
data,train_loader=prepare_train_loader(adata)
cost_time_prepare_data=time.time()-st2
print("Prepare train loader cost time : {}".format(cost_time_prepare_data))
################################################################################
st3=time.time()
trainer = Trainer(input_dims=data.num_features)
trainer.train(train_loader=train_loader, epochs=epochs)
feat = trainer.inference(train_loader, adata.shape[0])
cost_time_train=time.time()-st3
print("Train cost time : {}".format(cost_time_train))
################################################################################
st4=time.time()
cost_time_cluster,results = cluster_block(feat=feat, adata=adata, indices=list(range(feat.shape[0])),
                            save_path=save_path, n_neighbors=30, resolution=0.5)
cost_time_cluster=time.time()-st4
print("Cluster cost time : {}".format(cost_time_cluster))
################################################################################
params = {"cost_time_prepare_data": str(cost_time_prepare_data), 
          "cost_time_train": str(cost_time_train), 
          "cost_time_cluster": str(cost_time_cluster)}



df_concat=pd.concat(dfs)
df_concat["id"] =df_concat["Tag"].map(str) +"_"+ df_concat["Label"].map(str)
df_pos=df_concat.groupby("id").mean()[["x", "y","Tag","Label"]]
df_pos.sort_values(by=["Tag","Label"], inplace=True, ascending=True,ignore_index=True)
df_cluster=pd.read_csv("./result/cluster_batch.csv",sep=",")
df_pos["type"]=df_cluster["label"]
color_dict={}
for i in range(0,20):
    color_dict[i]=[plt.cm.tab20(int(i))]
color_list=[color_dict[i] if i in color_dict else i for i in df_pos["type"]]
plt.figure(figsize=(10,8))
# df_cluster=pd.read_csv("/mnt/13d1/ganshuang/jupyter_notebook/result/cluster.csv",sep="\t")
plt.scatter(df_pos["x"],df_pos["y"], c=color_list, cmap='Spectral',s=1.5)
plt.title('cluster of brain')
plt.savefig("./result/cluster_plot.png")
