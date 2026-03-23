import csv
import random
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import dgl

from scipy import sparse as sp
from collections import OrderedDict


class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        # 简单的 CSV 日志器：按列写入训练/验证过程中的指标
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    # 统计模型参数量（用于打印/对比模型规模）
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    # 导出模型结构与参数量明细（可选保存到文件）
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) + \
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def common_loss(emb1, emb2):
    # AdaDR 原始的双视图一致性损失（不是对比学习）：
    # 1) 去均值 + L2 归一化
    # 2) 计算两者的“样本-样本相似度矩阵”(cov) 并最小化差异
    # 直观理解：让两种视图的几何结构尽量一致
    emb1 = emb1 - th.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - th.mean(emb2, dim=0, keepdim=True)
    emb1 = th.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = th.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = th.matmul(emb1, emb1.t())
    cov2 = th.matmul(emb2, emb2.t())
    cost = th.mean((cov1 - cov2) ** 2)
    return cost


def setup_seed(seed):
    # 固定随机种子，尽量保证实验可复现
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


def knn_graph(sim, k):
    """
    Construct a k-nearest neighbor graph from a similarity matrix.
    Args:
        sim: Similarity matrix (numpy array or torch tensor)
        k: Number of neighbors
    Returns:
        adj: Sparse adjacency matrix of the KNN graph
    """
    # 说明：这里 sim 是“相似度矩阵”（越大越相似），因此取每行 top-k 最大值作为邻居
    # 返回的 adj 是 0/1 稀疏邻接矩阵（不带权），并做对称化以保证无向图结构
    if isinstance(sim, th.Tensor):
        sim = sim.cpu().numpy()
    
    n = sim.shape[0]
    if k > n:
        k = n
        
    # Get the indices of the k largest elements for each row
    # We use argpartition for efficiency. -k puts the k largest elements at the end.
    ind = np.argpartition(sim, -k, axis=1)[:, -k:]
    
    # Create the adjacency matrix
    row_indices = np.repeat(np.arange(n), k)
    col_indices = ind.flatten()
    data = np.ones(n * k)
    
    adj = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
    # Symmetrize the graph (Critical fix from AdaDR original)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # features: [batch_size, hidden_dim]
        # features contains [view1; view2] concatenated
        # 约定：features 前半部分是 view1，后半部分是 view2，
        # 同一索引 i 的两视图 (i, i+N) 互为正样本，其余为负样本
        batch_size = features.shape[0] // 2
        
        # Normalize features
        features = nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = th.matmul(features, features.T)
        
        # Mask out self-contrast
        mask = th.eye(2 * batch_size).to(features.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Select positives
        # view1[i] <-> view2[i] are positive pairs
        # indices: 0..N-1 are view1, N..2N-1 are view2
        # pos for 0 is N, pos for 1 is N+1...
        
        # Create labels: for row i (0..N-1), label is i+N
        # for row i (N..2N-1), label is i-N
        labels = th.cat([th.arange(batch_size) + batch_size, th.arange(batch_size)]).to(features.device)
        
        # Calculate logits
        logits = similarity_matrix / self.temperature
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 二分类 Focal Loss（基于 BCEWithLogits）：
        # - alpha：控制正负样本的权重
        # - gamma：控制“难样本挖掘”强度，gamma 越大越关注错分样本
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = th.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def mask_node_features(features, dropout_prob):
    """
    Randomly mask node features with zeros.
    """
    # 节点特征增强：随机把部分特征位置置 0（类似 dropout on features）
    if dropout_prob == 0:
        return features
    
    mask = th.rand(features.shape) > dropout_prob
    return features * mask.to(features.device)

def random_edge_dropout(graph, dropout_prob):
    """
    Randomly drop edges from a graph.
    Supports dgl.DGLGraph and torch.sparse.FloatTensor.
    """
    # 边增强：随机丢弃一部分边，形成另一个图视图
    if dropout_prob == 0:
        return graph

    if isinstance(graph, dgl.DGLGraph):
        num_edges = graph.num_edges()
        ids_to_remove = th.nonzero(th.rand(num_edges) < dropout_prob).squeeze()
        if ids_to_remove.numel() > 0:
            ids_to_remove = ids_to_remove.to(graph.device)
            new_graph = dgl.remove_edges(graph, ids_to_remove)
            return new_graph
        return graph
    
    elif isinstance(graph, th.Tensor) and graph.is_sparse:
        indices = graph._indices()
        values = graph._values()
        num_edges = indices.shape[1]
        
        mask = th.rand(num_edges) > dropout_prob
        
        new_indices = indices[:, mask]
        new_values = values[mask]
        
        new_graph = th.sparse_coo_tensor(new_indices, new_values, graph.size(), device=graph.device)
        return new_graph
        
    else:
         # Fallback for other types or error
         # If it is a dense tensor, maybe we can ignore or warn
         return graph
