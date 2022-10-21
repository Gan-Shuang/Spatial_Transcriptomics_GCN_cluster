# ST_GCN_cluster
Spatial Transcriptomics Cluster
## Introduction
WORKFLOW  
1、transform Spatial Transcriptomics to anndata  
2、PCA  
3、VAE & VGAE with GCN
4、Leiden cluster  
REFERENCE  
Hu J., Li X., Coleman K., Schroeder A., Ma N., Irwin D.J., Lee E.B., Shinohara R.T., Li M. SpaGCN: integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network. Nat. Methods. 2021; 18:1342–1351.  
Fu H., Xu H., Chong K., Li M., Ang K.S., Lee H.K., Ling J., Chen A., Shao L., Liu L. et al. . Unsupervised spatially embedded deep representation of spatial transcriptomics. 2021; bioRxiv doi:16 June 2021, preprint: not peer reviewedhttps://doi.org/10.1101/2021.06.15.448542.  
Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. 2019. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19). Association for Computing Machinery, New York, NY, USA, 257–266. https://doi.org/10.1145/3292500.3330925  

## Method
```
cd ST_GCN_cluster
unset DISPLAY
python GCN_leiden.py
```
## Results
> cluster_batch.csv
```
id,umap_x,umap_y,label
```

> cluster_plot.png

![brain](https://user-images.githubusercontent.com/50703435/197137548-f92488c7-0f44-43e7-80a6-754addf54f45.png)

![cluster4](https://user-images.githubusercontent.com/50703435/197137818-dee56c50-6fc1-4cc7-a4e6-4a22b8db3d4b.png)

