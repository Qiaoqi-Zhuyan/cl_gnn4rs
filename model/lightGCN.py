import torch
from torch import nn


class LightGCN(nn.Module):
    """
    lightGCN 实现
    """
    def __init__(self, user_num, item_num, embed_dim, n_layer):
        super(LightGCN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.n_layer = n_layer

        self.user_embed = nn.Embedding(self.user_num, self.embed_dim)
        self.item_embed = nn.Embedding(self.item_num, self.embed_dim)

    def reset_params(self):
        init = torch.nn.init.xavier_uniform_
        init(self.user_embedding.weight)
        init(self.item_embedding.weight)

    def forward(self, norm_adj):
        ego_embed = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
        all_embed = [ego_embed]

        for k in range(self.n_layer):
            ego_embed = torch.sparse.mm(norm_adj, ego_embed)
            all_embed += [ego_embed]

        all_embed = torch.stack(all_embed, dim=1).mean(dim=1)
        user_embedding, item_embedding = torch.split(all_embed, [self.user_num, self.item_num], dim=0)

        return user_embedding, item_embedding



