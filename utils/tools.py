import numpy as np

import torch


def sp_mat_to_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


def inner_product(a, b):
    return torch.sum(a*b, dim=-1)


def evaluate_metric(ratings,test_data ,k):
    hit = 0
    DCG = 0.
    iDCG = 0.

    _, shoot_index = torch.topk(ratings, k)
    shoot_index = shoot_index.cpu().tolist()

    for i in range(len(shoot_index)):
        if shoot_index[i] in test_data:
            hit += 1
            DCG += 1 / np.log2(i + 2)
        if i < test_data.size()[0]:
            iDCG += 1 / np.log2(i + 2)

    recall = hit / test_data.size()[0]
    NDCG = DCG / iDCG

    return recall, NDCG