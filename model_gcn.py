import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import torch
import time

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            print("this is a node layer")
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            print("this is an edge layer")
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t()
            mask1 = torch.eye(multiplier1.shape[0])
            M1 = mask1 * torch.ones(multiplier1.shape[0]) + (1. - mask1)*multiplier1
            adjusted_A = torch.mul(M1, adj_v.to_dense())
            '''
            print("adjusted_A is ", adjusted_A)
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            print("normalized adjusted A is ", normalized_adjusted_A)
            '''
            # to avoid missing feature's influence, we don't normalize the A
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e

        else:
            multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T.to_dense()
            mask2 = torch.eye(multiplier2.shape[0])
            M3 = mask2 * torch.ones(multiplier2.shape[0]) + (1. - mask2)*multiplier2
            adjusted_A = torch.mul(M3, adj_e.to_dense())
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, nclass, dropout, node_layer=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nclass, nfeat_e, nfeat_e, node_layer=True)
        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T, pooling=1, node_count=1, graph_level=True):
        # print x
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc3(X, Z, adj_e, adj_v, T)
        return F.log_softmax(X, dim=1)