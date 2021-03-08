import torch

def get_gramm_matrix(feature):
    B, C, H, W = feature.size()
    feature = feature.flatten(2)
    return torch.matmul(feature, feature.transpose(1, 2)) / (C * H * W)