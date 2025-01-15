# -*- coding: UTF-8 -*-
# @Author  : Based on ICLR 2023 LightGCL Paper
# @Email   : Example implementation for ReChorus

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class LightGCLBase(object):
    """
    Base class for LightGC model, providing methods for argument parsing,
    adjacency matrix construction, and SVD approximation.
    """
    
    @staticmethod
    def parse_model_args(parser):
        """
        Parse command-line arguments specific to LightGCL model.
        
        Args:
            parser: Argument parser object to add model-specific arguments.
        
        Returns:
            parser: Updated parser object.
        """
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of LightGCL layers.')
        parser.add_argument('--svd_rank', type=int, default=5,
                            help='Rank for SVD-based augmentation.')
        return parser

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat):
        """
        Build the adjacency matrix from the training interaction data.
        
        Args:
            user_count (int): Number of users.
            item_count (int): Number of items.
            train_mat (dict): Training data where keys are user ids and values are sets of item ids.

        Returns:
            adj_mat (scipy.sparse.csr_matrix): Constructed adjacency matrix in CSR format.
        """
        R = sp.lil_matrix((user_count, item_count))  # 使用 LIL 格式创建稀疏矩阵

        # 遍历每个用户和他们点击的物品
        for user, items in train_mat.items():
            for item in items:
                R[user, item] = 1  # 假设每个 (user, item) 对应一个交互，设置为1表示点击/评分

        # 创建一个大小为 (user_count + item_count) x (user_count + item_count) 的邻接矩阵
        adj_mat = sp.lil_matrix((user_count + item_count, user_count + item_count))  
        adj_mat[:user_count, user_count:] = R  # 将用户与物品之间的交互矩阵填充到邻接矩阵的左上角
        adj_mat[user_count:, :user_count] = R.T  # 将 R 的转置填充到邻接矩阵的右上角

        return adj_mat

    @staticmethod
    def approximate_svd(adj_mat, rank):
        """
        Perform Singular Value Decomposition (SVD) approximation on the adjacency matrix.
        
        Args:
            adj_mat (scipy.sparse.csr_matrix): Adjacency matrix to decompose.
            rank (int): Rank for the SVD decomposition.
        
        Returns:
            u (scipy.sparse.csr_matrix): Left singular vectors.
            s_diag (scipy.sparse.csr_matrix): Singular values as a diagonal matrix.
            vt (scipy.sparse.csr_matrix): Right singular vectors.
        """
        u, s, vt = sp.linalg.svds(adj_mat, k=rank)
        s_diag = sp.diags(s)
        return u, s_diag, vt

    def _base_init(self, args, corpus):
        """
        Initialize base parameters and model components.
        
        Args:
            args: Model arguments.
            corpus: Dataset object containing user-item interaction data.
        """
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.svd_rank = args.svd_rank
        
        # Construct normalized adjacency matrix
        self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        
        # Approximate the SVD of the adjacency matrix
        self.svd_u, self.svd_s, self.svd_vt = self.approximate_svd(self.norm_adj, self.svd_rank)
        
        # Define the encoder model
        self._base_define_params()
        
        # Apply weight initialization
        self.apply(self.init_weights)

    def _base_define_params(self):
        """
        Define the model parameters and the LightGCLEncoder.
        """
        self.encoder = LightGCLEncoder(self.user_num, self.item_num, self.emb_size, 
                                       self.norm_adj, self.svd_u, self.svd_s, self.svd_vt, 
                                       self.n_layers)

    def forward(self, feed_dict):
        """
        Perform forward pass through the model.
        
        Args:
            feed_dict (dict): Dictionary containing 'user_id' and 'item_id' for prediction.

        Returns:
            dict: Dictionary containing the prediction tensor.
        """
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)

        # Compute the prediction (dot product between user and item embeddings)
        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
        return {'prediction': prediction}

class LightGCL(GeneralModel, LightGCLBase):
    """
    LightGC model combining GeneralModel and LightGCLBase.
    """

    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'svd_rank']

    @staticmethod
    def parse_model_args(parser):
        """
        Parse command-line arguments specific to LightGCL model.
        
        Args:
            parser: Argument parser object to add model-specific arguments.
        
        Returns:
            parser: Updated parser object.
        """
        parser = LightGCLBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        """
        Initialize the LightGCL model.
        
        Args:
            args: Model arguments.
            corpus: Dataset object containing user-item interaction data.
        """
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        """
        Forward pass through the LightGCL model.
        
        Args:
            feed_dict (dict): Dictionary containing 'user_id' and 'item_id' for prediction.
        
        Returns:
            dict: Dictionary containing the prediction tensor.
        """
        return LightGCLBase.forward(self, feed_dict)

class LightGCLEncoder(nn.Module):
    """
    Encoder for the LightGC model, implementing the light collaborative filtering layers.
    """

    def __init__(self, user_count, item_count, emb_size, norm_adj, svd_u, svd_s, svd_vt, n_layers=3):
        """
        Initialize the encoder parameters and setup the embedding layers.
        
        Args:
            user_count (int): Number of users.
            item_count (int): Number of items.
            emb_size (int): Size of embedding vectors.
            norm_adj (scipy.sparse.csr_matrix): Normalized adjacency matrix.
            svd_u (torch.Tensor): Left singular vectors from SVD.
            svd_s (torch.Tensor): Singular values from SVD.
            svd_vt (torch.Tensor): Right singular vectors from SVD.
            n_layers (int): Number of collaborative filtering layers.
        """
        super(LightGCLEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.svd_u = torch.tensor(svd_u, dtype=torch.float32)
        svd_s_dense = svd_s.toarray()  # 将 dia_matrix 转换为密集的 NumPy 数组
        self.svd_s = torch.tensor(svd_s_dense, dtype=torch.float32)  # 转换为 PyTorch 张量
        self.svd_vt = torch.tensor(svd_vt, dtype=torch.float32)

        # Initialize user and item embeddings
        self.embedding_dict = self._init_model()

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to("cpu")  # 或者使用 .to(device) 根据 device 设置


    def _init_model(self):
        """
        Initialize user and item embeddings using Xavier initialization.
        
        Returns:
            embedding_dict (nn.ParameterDict): Dictionary containing user and item embeddings.
        """
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        """
        Convert a sparse matrix to a sparse tensor format for PyTorch.
        
        Args:
            X (scipy.sparse.csr_matrix): Sparse matrix to convert.
        
        Returns:
            torch.sparse.FloatTensor: Converted sparse tensor.
        """
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, users, items):
        """
        Perform forward pass for user and item embeddings through the LightGCLEncoder layers.
    
        Args:
            users (torch.Tensor): Tensor of user IDs.
            items (torch.Tensor): Tensor of item IDs.
    
        Returns:
            user_embeddings (torch.Tensor): User embeddings.
            item_embeddings (torch.Tensor): Item embeddings.
        """
        # Concatenate user and item embeddings
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        # Propagate through layers
        for k in range(self.n_layers):
            # Apply sparse matrix multiplication
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Average embeddings from all layers
        all_embeddings = torch.stack(all_embeddings, dim=0)  # Stack all layers' embeddings
        all_embeddings = all_embeddings.mean(dim=0)  # Average over layers

        # Extract user and item embeddings
        user_embeddings = all_embeddings[:self.user_count]  # Extract user embeddings
        item_embeddings = all_embeddings[self.user_count:]  # Extract item embeddings

        # Retrieve specific user and item embeddings
        users = users.long()
        items = items.long()
        u_embed = user_embeddings[users]
        i_embed = item_embeddings[items]

        # Return the embeddings as a tuple
        return u_embed, i_embed