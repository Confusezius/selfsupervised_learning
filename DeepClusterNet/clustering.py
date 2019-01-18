"""=================================================================="""
"""=================== LIBRARIES ===================================="""
"""=================================================================="""
import os, sys, numpy as np, time
from tqdm import tqdm, trange
import torch, torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import faiss, matplotlib.pyplot as plt
import auxiliaries as aux
import umap






"""========================================================================================================"""
### Top function holder which runs cluster computation of network features, optional preprocessing using PCA,
### assigning clusters to dataloader as new targets and adjusts the network to the number of clusters accordingly.
def compute_clusters_and_set_dataloader_labels(dataloader, cluster_network, opt):
    image_clusters, losses = generate_cluster_labels(dataloader, cluster_network, opt)
    pseudolabels = adjust_dataloader_labels(dataloader, image_clusters)
    return pseudolabels





"""========================================================================================================"""
### Using the network features (<aux.prepare_model_4_clustering>),
### compute features for all training images and find clusters, setting them as new training targets (<compute_clusters_and_set_dataloader_labels>).
### Finally, add back the top layer to the network with number_output_units==number_of_clusters
def adjust_model_compute_clusters_and_set_dataloader_labels(image_dataloader, model, opt, logs):
    end = time.time()

    aux.prepare_model_4_clustering(model)
    # model = prepare_model_4_clustering(model)

    image_dataloader.dataset.labels = None
    pseudolabels = compute_clusters_and_set_dataloader_labels(image_dataloader, model, opt)

    # model = rebuild_model_4_training(model, len(pseudolabels))
    aux.rebuild_model_4_training(model, opt.num_cluster)
    logs.logs['cluster time'].update(end-time.time())




"""========================================================================================================"""
### Function holder which runs feature computation on training images <compute_features>, optional PCA-preprocessing,
### clustering <cluster> and optional umap generation.
def generate_cluster_labels(dataloader, cluster_network, opt):
    features, labels = compute_features(dataloader, cluster_network, opt)
    if opt.use_dim_reduction:
        features = feature_preprocessing(features, opt.pca_dim)
    image_clusters, losses = cluster(features, opt.num_cluster)
    if opt.make_umap_plots:
        generate_umap(features, labels, image_clusters, label_idx=15)
    return image_clusters, losses




"""========================================================================================================"""
### Using the network.features of choice, run through the full dataset and compute feature vectors for all
### input images. Collect them and pass them to <feature_preprocessing> and/or <cluster> for clustering.
### If the make_umap_plots-flag is set, the additional label vector is saved.
def compute_features(dataloader, cluster_network, opt):
    cluster_network.eval()
    dataloader.dataset.random_sample = False

    iterator = tqdm(dataloader, position=1)
    iterator.set_description('Computing features & Clustering... ')

    feature_coll, label_coll = [],[]
    for i,file_dict in enumerate(iterator):
        input_image = file_dict['Input Image'].to(opt.device)
        feature     = cluster_network(input_image).detach().cpu().numpy()
        feature_coll.append(feature.astype('float32'))
        if opt.make_umap_plots: label_coll.extend(list(file_dict['Label Vector'].numpy()))

    return np.vstack(feature_coll), np.vstack(label_coll) if opt.make_umap_plots else None





"""========================================================================================================"""
### Run optional feature preprocessing to select the <pca_dim> most important features for quicker cluster search.
def feature_preprocessing(features, pca_dim=256):
    _, feature_dim   = features.shape
    features = features.astype('float32')

    PCA_generator = faiss.PCAMatrix(feature_dim, pca_dim, eigen_power=-0.5)
    PCA_generator.train(features)

    # PCA-Whitening
    assert PCA_generator.is_trained
    features = PCA_generator.apply_py(features)

    # L2 normalization
    row_sum = np.linalg.norm(features, axis=1)
    featrues = features/row_sum[:, np.newaxis]

    return features



"""========================================================================================================"""
### Cluster the set of feature vectors passed from <compute_features> using faiss.Clustering().
### The number of clusters if given by <num_cluster>. Clusters are measured via L2-distance.
### <image_list> is passed as <image_clusters>  in <compute_clusters_and_set_dataloader_labels> to
### <adjust_dataloader_labels> to be set as new training labels in the provided dataset.
def cluster(features, num_cluster):
    n_samples, dim = features.shape

    kmeans_clustering = faiss.Clustering(dim, num_cluster)
    kmeans_clustering.n_iter = 20
    kmeans_clustering.max_points_per_centroid = 1000000000

    gpu_resource = faiss.StandardGpuResources()
    gpu_flat     = faiss.GpuIndexFlatConfig()
    gpu_flat.useFloat16 = False
    gpu_flat.device     = 0

    gpu_distance_measure = faiss.GpuIndexFlatL2(gpu_resource, dim, gpu_flat)


    kmeans_clustering.train(features, gpu_distance_measure)
    _, cluster_idxs  = gpu_distance_measure.search(features, 1)
    losses = faiss.vector_to_array(kmeans_clustering.obj)

    image_list = [[] for i in range(num_cluster)]
    for i in range(len(features)):
        image_list[cluster_idxs[i][0]].append(i)

    return image_list, losses[-1]




"""========================================================================================================"""
### Optionally, generate an umap visalization of features and their respective cluster labels. This can take
### quite long for full datasets and is suggested to be performed separately after <x> epochs.
def generate_umap(features, labels, image_clusters, label_idx):
    label_idx_labels = np.where(labels[:,label_idx]==1)[0]
    non_label_idx_labels = [i for i in range(len(features)) if i not in label_idx_labels]

    umap_base = umap.UMAP(n_components=2)
    mapped_features = umap_base.fit_transform(features)
    cluster_colors = np.zeros(len(features))
    for i,cluster in enumerate(image_clusters):
        for cluster_idx in cluster:
            cluster_colors[cluster_idx]=i
    f,ax = plt.subplots(1,2)
    ax[0].scatter(mapped_features[non_label_idx_labels,0],mapped_features[non_label_idx_labels,1],c='r',s=0.2)
    ax[0].scatter(mapped_features[label_idx_labels,0],mapped_features[label_idx_labels,1],c='g',s=0.8)
    ax[0].set_title('Mapping of exemplary label')
    ax[1].scatter(mapped_features[:,0], mapped_features[:,1], c=cluster_colors, s=0.2)
    ax[1].set_title('Coloured Cluster')
    f.set_size_inches(15,8)
    f.savefig(opt.savefolder+'/umap_'+str(opt.epoch)+'.png')
    plt.close()



"""========================================================================================================"""
### Using the computed image clusters in <cluster>/<generate_cluster_labels> respectively, we adjust the
### Dataloader training labels.
def adjust_dataloader_labels(dataloader, image_clusters):
    pseudolabels, image_idxs = [],[]
    for cluster_idx,image_cluster in enumerate(image_clusters):
        image_idxs.extend(image_cluster)
        pseudolabels.extend([cluster_idx]*len(image_cluster))

    label_to_idx = {label: idx for idx,label in enumerate(set(pseudolabels))}

    sort_idxs = np.argsort(image_idxs)
    pseudolabels = np.array(pseudolabels)[sort_idxs]

    # Handle empty image_clusters
    pseudolabels = [label_to_idx[pseudolabel] for pseudolabel in pseudolabels]

    dataloader.dataset.labels = pseudolabels
    return pseudolabels
