import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from torch import optim

import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from utils.dataset_loader import *
from utils.tools import *
from utils.loss import *
from model.lightGCN import LightGCN

import os

class SGL(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data_path = args.train_data_path
        self.test_data_path  = args.test_data_path

        # 加载数据集
        train_data = pd.read_csv(self.train_data_path, sep=',', header=None, names=['user', 'item'])
        test_data  = pd.read_csv(self.test_data_path, sep=',', header=None, names=['user', 'item'])
        all_data = pd.concat([train_data, test_data])
        self.user_num = max(all_data['user']) + 1
        self.item_num = max(all_data['item']) + 1

        self.train_dataset = Dataset_train(train_data, self.user_num, self.item_num)
        self.train_loader = dataloader.DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        self.test_dataset = Dataset_test(test_data)
        self.test_loader = dataloader.DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # 图构建参数
        self.dropout_ratio = args.dropout_ratio
        self.graph = sp_mat_to_tensor(self.create_adj_norm(is_subgraph=False)).to(self.device)

        # 模型参数
        self.embed_dim = args.embed_dim
        self.layer_num = args.layer_num
        self.model = LightGCN(self.user_num, self.item_num, self.embed_dim, self.layer_num).to(self.device)
        self.cl_reg = args.cl_reg
        self.reg = args.reg

        # 训练参数
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.dataset = args.datadir

        # 评价指标
        self.k = args.k
        self.tao = args.tao

        # 记录
        self.epoch = 0
        self.cnt = 0
        self.train_loss = []
        self.bpr_loss = []
        self.infoNCE_loss = []
        self.reg_loss = []
        self.recall_history = []
        self.NDCG_history = []
        self.best_recall = 0
        self.best_NDCG = 0
        self.best_epoch = 0

    def train_one_epoch(self):

        epoch_criterion = 0.
        epoch_bpr_criterion = 0.
        epoch_infonce_criterion = 0.
        epoch_reg_criterion = 0.

        sub_graph1 = sp_mat_to_tensor(self.create_adj_norm(is_subgraph=True, aug_type='ed')).to(self.device)
        sub_graph2 = sp_mat_to_tensor(self.create_adj_norm(is_subgraph=True, aug_type='ed')).to(self.device)

        self.model.train()
        for batch_user, batch_pos_item, batch_neg_item in tqdm(self.train_loader):
            batch_user = batch_user.long().to(self.device)
            batch_pos_item = batch_pos_item.long().to(self.device)
            batch_neg_item = batch_neg_item.long().to(self.device)

            all_user_embed, all_item_embed = self.model(self.graph)
            sub_graph_user_embed1, sub_graph_item_embed1 = self.model(sub_graph1)
            sub_graph_user_embed2, sub_graph_item_embed2 = self.model(sub_graph2)

            # 进行归一化
            sub_graph_user_embed1 = F.normalize(sub_graph_user_embed1)
            sub_graph_item_embed1 = F.normalize(sub_graph_item_embed1)
            sub_graph_user_embed2 = F.normalize(sub_graph_user_embed2)
            sub_graph_item_embed2 = F.normalize(sub_graph_item_embed2)

            # 选择batch中的用户-物品
            batch_user_embed = all_user_embed[batch_user]
            batch_pos_item_embed = all_item_embed[batch_pos_item]
            batch_neg_item_embed = all_item_embed[batch_neg_item]
            batch_sub_user_embed1 = sub_graph_user_embed1[batch_user]
            batch_sub_user_embed2 = sub_graph_user_embed2[batch_user]
            batch_sub_item_embed1 = sub_graph_item_embed1[batch_pos_item]
            batch_sub_item_embed2 = sub_graph_item_embed2[batch_pos_item]

            pos_score = inner_product(batch_user_embed, batch_pos_item_embed) #[batch_size]
            neg_score = inner_product(batch_user_embed, batch_neg_item_embed)

            sub_graph_user_pos_score = inner_product(batch_sub_user_embed1, batch_sub_user_embed2)  # [batch_size]
            sub_graph_user_neg_score = torch.matmul(batch_sub_user_embed1, torch.transpose(batch_sub_user_embed2, 0, 1))  # [batch_size, num_user]

            sub_graph_item_pos_score = inner_product(batch_sub_item_embed1, batch_sub_item_embed2)  # [batch_size]
            sub_graph_item_neg_score = torch.matmul(batch_sub_item_embed1, torch.transpose(batch_sub_item_embed2, 0, 1))  # [batch_size, num_item]

            # bpr loss (main loss)
            bpr_criterion = bpr_loss(pos_score, neg_score)

            # contrastive learning
            user_infonce_loss = infonce_loss(sub_graph_user_pos_score, sub_graph_user_neg_score, self.tao)
            item_infonce_loss = infonce_loss(sub_graph_item_pos_score, sub_graph_item_neg_score, self.tao)
            infonce_criterion = torch.sum(user_infonce_loss + item_infonce_loss, dim=-1)

            # l2 regularization
            reg_loss = l2_loss(self.model.user_embed(batch_user), self.model.item_embed(batch_pos_item), self.model.item_embed(batch_neg_item))

            loss = bpr_criterion + self.cl_reg * infonce_criterion + self.reg * reg_loss

            epoch_criterion += loss
            epoch_bpr_criterion += bpr_criterion
            epoch_infonce_criterion += self.cl_reg * infonce_criterion
            epoch_reg_criterion += self.reg * reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_criterion, epoch_bpr_criterion, epoch_infonce_criterion, epoch_reg_criterion

    def create_adj_norm(self, is_subgraph=False, aug_type='ed'):
        node_num = self.user_num + self.item_num
        user_list, item_list = self.train_dataset.user_idx, self.train_dataset.item_idx

        if is_subgraph:

            if aug_type == "nd":
                # 去除某些节点
                # user node
                user_sample_size = int(self.user_num * self.dropout_ratio)
                drop_user_idx = np.array(self.user_num)
                np.random.shuffle(drop_user_idx)
                drop_user_idx = drop_user_idx[:user_sample_size]
                indicator_user = np.ones(self.user_num, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)

                # item node
                item_sample_size = int(self.item_num * self.dropout_ratio)
                drop_item_idx = np.array(self.item_num)
                np.random.shuffle(drop_item_idx)
                drop_item_idx = drop_item_idx[:item_sample_size]
                indicator_item = np.ones(self.item_num, dtype=np.float32)
                indicator_item[drop_item_idx] = 0.
                diag_indicator_item = sp.diags(indicator_item)

                R = sp.csr_matrix((np.ones_like(user_list, dtype=np.float32), (user_list, item_list)), shape=(self.user_num, self.item_num))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_list_keep, item_list_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_list_keep, item_list_keep + self.user_num)), shape=(node_num, node_num))

            if aug_type in ['ed', 'rw']:
                # 邻接矩阵，随机选取边删除
                sample_size = int(user_list.shape[0] * (1 - self.dropout_ratio))
                keep_idx = np.arange(user_list.shape[0])
                np.random.shuffle(keep_idx)
                keep_idx = keep_idx[:sample_size]
                user_list = np.array(user_list)[keep_idx]
                item_list = np.array(item_list)[keep_idx]
                ratings = np.ones_like(user_list)
                tmp_adj = sp.csr_matrix((ratings, (user_list, item_list + self.user_num)), shape=(node_num, node_num))
        else:
            rating = np.ones_like(user_list, dtype=np.float32)
            tmp_adj = sp.csr_matrix((rating, (user_list, item_list + self.user_num)), shape=(node_num, node_num))

        adj_mat = tmp_adj + tmp_adj.T

        row_sum = np.array(adj_mat.sum(1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def test_epoch(self):
        test_user_pos_dict = self.test_dataset.user_pos_dict
        train_user_pos_dict = self.train_dataset.user_pos_dict

        epoch_recall = 0.
        epoch_NDCG = 0.
        tot = 0.

        self.model.eval()
        for test_user in self.test_loader:
            user_num = test_user.size()[0]
            test_user = test_user.long().to(self.device)
            test_item = [torch.from_numpy(test_user_pos_dict[int(u)]).long().to(self.device) for u in test_user]

            all_user_embed, all_item_embed = self.model(self.graph)
            test_user_embed = all_user_embed[test_user]
            ratings = torch.matmul(test_user_embed, all_item_embed.T)

            for idx, user in enumerate(test_user):
                train_items = train_user_pos_dict[int(user.cpu())]
                ratings[idx][train_items] = -np.inf

            for i in range(user_num):
                recall, NDCG = evaluate_metric(ratings[i], test_item[i], self.k)
                epoch_recall += recall
                epoch_NDCG += NDCG

            tot += user_num

        epoch_recall /= tot
        epoch_NDCG /= tot

        return epoch_recall, epoch_NDCG

    def run(self):
        for epoch in tqdm(range(1, self.epoch_num + 1)):
            epoch_criterion, epoch_bpr_criterion, epoch_infonce_criterion, epoch_reg_criterion = self.train_one_epoch()
            self.train_loss.append(epoch_criterion)
            self.bpr_loss.append(epoch_bpr_criterion)
            self.infoNCE_loss.append(epoch_infonce_criterion)
            self.reg_loss.append(epoch_reg_criterion)

            # print(f"Epoch {self.epoch}:  loss:{epoch_criterion/self.train_dataset.interact_num} \
            #         bpr_loss:{epoch_bpr_criterion/self.train_dataset.interact_num} \
            #         info_NCE_loss:{epoch_infonce_criterion/self.train_dataset.interact_num} \
            #         reg_loss:{epoch_reg_criterion/self.train_dataset.interact_num}")


            print("Epoch : %d /%d  \
                    loss: %.4f \
                    bpr_loss: %.4f \
                    info_NCE_loss: %.4f \
                    reg_loss: %.4f" %
                  (epoch,
                  self.epoch_num,
                  epoch_criterion / self.train_dataset.interact_num,
                  epoch_bpr_criterion / self.train_dataset.interact_num,
                  epoch_infonce_criterion / self.train_dataset.interact_num,
                  epoch_reg_criterion / self.train_dataset.interact_num)
                  )

            if epoch % 20 == 0:
                epoch_recall, epoch_NDCG = self.test_epoch()
                self.recall_history.append(epoch_recall)
                self.NDCG_history.append(epoch_NDCG)
                print("recall@%d: %.4f, NDCG@%d: %.4f" % (self.k, epoch_recall, self.k, epoch_NDCG))

        # 保存模型
        folder = './runs/' + self.time + '/' + str(self.epoch_num) + '/'
        fname = 'sgl.epochs={}.lr={}.layer={}.batch_size={}.dataset={}.pth'
        fname = fname.format(self.epoch_num, self.lr, self.layer_num, self.batch_size, self.dataset)
        torch.save(self.model.state_dict(), os.path.join(folder, fname))
        self.save_metrics(folder)

    def save_metrics(self, path):
        writer = SummaryWriter(path)
        for i in range(self.epoch_num):
            writer.add_scalar('Loss', self.train_loss[i], i)
            writer.add_scalar('bpr_loss', self.bpr_loss[i], i)
            writer.add_scalar('infoNCE_loss', self.infoNCE_loss[i], i)
            writer.add_scalar('reg_loss', self.reg_loss[i], i)
            writer.add_scalar('Recall@20', self.recall_history[i], i)
            writer.add_scalar('NDCG@20', self.NDCG_history[i], i)


