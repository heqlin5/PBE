#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu


#regdb
# rerank_dist_all = compute_modal_camera_invariant_jaccard_distance(feature_all, k1=38, k2=18,
#                                                                   file=sorted(dataset_rgb.train) + sorted(
#                                                                       dataset_ir.train),
#                                                                   search_option=3,
#                                                                   camera_num=1)  # rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
# pseudo_labels_all = cluster_all.fit_predict(rerank_dist_all)


##sysu
# rerank_dist_all = compute_modal_camera_invariant_jaccard_distance(feature_all, k1=40, k2=32,
#                                                                   file=sorted(dataset_rgb.train) + sorted(
#                                                                       dataset_ir.train),
#                                                                   search_option=3,
#                                                                   camera_num=6)  # rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
# pseudo_labels_all = cluster_all.fit_predict(rerank_dist_all)
def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]




def compute_modal_camera_invariant_jaccard_distance(target_features, file, k1=20, k2=6, print_flag=True, search_option=0,
                                              use_float16=False,camera_num=9):
    end = time.time()
    all_file_name = []
    camera_id=[]
    for i, (fname, _, cid) in enumerate(file):
        all_file_name.append(fname)
        camera_id.append(cid)
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option == 0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option == 1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option == 2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2 - 2 * torch.mm(target_features[i].unsqueeze(0).contiguous(),
                                target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    print('modality-camera invariant expension')
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):



            feas_camera_ir_temp = [[] for _ in range(camera_num)]
            feas_camera_rgb_temp = [[] for _ in range(camera_num)]



            for ii in initial_rank[i, :k2]:
                camera_label = camera_id[ii]
                filename=all_file_name[ii]

                if 'ir_modify' in filename:

                    feas_camera_ir_temp[camera_label].append(V[ii, :])

                if 'rgb_modify' in filename:

                    feas_camera_rgb_temp[camera_label].append(V[ii, :])

            average_matices1 = []
            average_matices2 = []

            arrays_ir = [np.array(arr) for arr in feas_camera_ir_temp]
            arrays_rgb = [np.array(arr) for arr in feas_camera_rgb_temp]



            for idx, arr in enumerate(arrays_ir):
                if len(arr) > 0:
                    non_empty_rows = [row for row in arr if np.any(row)]

                    if non_empty_rows:
                        average_matrix = np.mean(non_empty_rows, axis=0)
                        average_matices1.append(average_matrix)

            for idx, arr in enumerate(arrays_rgb):
                if len(arr) > 0:
                    non_empty_rows = [row for row in arr if np.any(row)]

                    if non_empty_rows:
                        average_matrix = np.mean(non_empty_rows, axis=0)
                        average_matices2.append(average_matrix)



            if len(average_matices1)==0:
                average_matices2 = np.mean(average_matices2, axis=0)

                V_qe[i, :] = average_matices2
            elif len(average_matices2)==0:
                average_matices1 = np.mean(average_matices1, axis=0)

                V_qe[i, :] = average_matices1
            else:
                average_matices1 = np.mean(average_matices1, axis=0)
                average_matices2 = np.mean(average_matices2, axis=0)
                # print('3 average_matices1.shape', average_matices1.shape,average_matices2.shape)
                V_qe[i, :] = np.mean((average_matices1, average_matices2), axis=0)


        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist


