# SELF-SUPERVISED TRAINING PIPELINES
---

## Content
This folder contains implementations for self-supervised training, i.e. augmentation to the unsupervised training procedure by defining a data-based target task. These are ```JigsawNet``` based on _Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles_ (https://arxiv.org/abs/1603.09246) by _Noroozi et al._, and ```DeepClusterNet``` based on _Deep Clustering for Unsupervised Learning of Visual Features_ (https://arxiv.org/abs/1807.05520) by _Caron et al_.

The former partly uses parts given in https://github.com/bbrattoli/JigsawPuzzlePytorch and the later takes some elements from the original repository https://github.com/facebookresearch/deepcluster.

__NOTE:__ Although the Jigsaw converges to useful values, I haven't gotten my DeepCluster-Implementation to work, as it has a hard time converging. Hence this repo has at the moment primarily archival functions.

## Setup

```
SELFSUPERVISED (Jigsaw++, DeepClustering)
|
└───JigsawNet (Implementation based on Noroozi et al, brattoli git)
|   │   train.py - Main Training Script
|   │   baseline_conv.py/network_library.py - Basic network implementations
|   │   auxiliaries.py - auxiliary functions, including permutation computation.
|   │   CelebA_dataset_Jigsaw.py - PyTorch-dataset implementation for CelebA
|   |
|   └───Permutations (Folder of precomputated max. Hamming distance Permutations)
|   
└───DeepClusterNet (Implementation adapted from Repo)
|   │   train.py      - Main Training script
|   │   clustering.py - Holds clustering and target generation functions.
|   │   CelebA_dataset_DeepCluster.py - PyTorch-dataset implementation for CelebA
|   │   baseline_conv.py - Basic network implementations (AlexNet, MiniNet)
|   │   auxiliaries.py - auxiliary functions to run train.py
|   
└───Datasets (CelebA)
```

## Additional Information
Both pipelines run on the `CelebA`-dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) data which should be downloaded and extracted into the Datasets-folder.
