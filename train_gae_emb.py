from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import scipy.sparse as sp
import pandas as pd
import torch
from torch import optim

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, load_data_gi, mask_test_edges_gi, \
    mask_test_edges_cv, prepare_for_cv, mask_test_edges_gi_cv, mask_test_edges_cv_temp, create_double_features, \
    load_data_gi_emb

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='data_k562', help='type of dataset.')
parser.add_argument('--weighted', type=bool, default=False, help='weight')
parser.add_argument('--num', type=int, default=0, help='number of experiment')

args = parser.parse_args()


def save_emb(hidden_emb, nodes_names, output_path):
    data_emb = pd.DataFrame(hidden_emb, index=nodes_names)
    data_emb.to_csv(output_path)


def gae_generate_emb(args):
    best_model_path = f'best-model_{args.dataset}_{args.num}.pt'
    print("Using {} dataset".format(args.dataset))
    # adj, features = load_data(args.dataset)
    adj, features, nodes_names = load_data_gi_emb(args.dataset, weighted=args.weighted)
    n_nodes, feat_dim = features.shape

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj + sp.eye(adj.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        hidden_emb = mu.data.numpy()
        # roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
        #       "val_roc=", "{:.5f}".format(roc_curr),
        #       "val_ap=", "{:.5f}".format(ap_curr),
        #       "time=", "{:.5f}".format(time.time() - t)
        #       )
        # if roc_curr > best_val_auc:
        #     torch.save(model.state_dict(), best_model_path)
        #     best_val_auc = roc_curr

    print("Optimization Finished!")
    model.load_state_dict(torch.load(best_model_path))
    recovered, mu, logvar = model(features, adj_norm)
    hidden_emb = mu.data.numpy()
    save_emb(hidden_emb, nodes_names, os.path.join(args.dataset, "emb_features.csv"))


if __name__ == '__main__':
    gae_generate_emb(args)
