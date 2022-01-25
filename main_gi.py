from __future__ import division
from __future__ import print_function

import argparse
import pickle
from os.path import join

import numpy as np
import scipy.sparse as sp
import torch

from load_data import load_data_gi
from model import GCNModelVAE, GCNModelVAENorm, GCNModelVAE2
from train_model import train_model, evaluate_model
from utils import mask_test_edges_cv, prepare_for_cv, mask_test_edges_gi_cv, mask_test_edges_separate_genes, \
    compare_to_rf_onto, print_results, mask_test_edges_compare

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vaen', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden0', type=int, default=128, help='Number of units in gnae layer.')
parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
parser.add_argument('--patience', type=int, default=10, help='Patience before early stopping of the training')
parser.add_argument('--identity', action='store_true', help='initialize vaen model normalization layer to identityy layer')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='data_K562', help='type of dataset.')
parser.add_argument('--weighted', action='store_true', help='weight')
parser.add_argument('--num', type=int, default=0, help='number of experiment')
parser.add_argument('--nos', type=int, default=5, help='Number of splits for CV')
parser.add_argument('--all_neu', action='store_true', help='Include all neutral samples - data is unbalanced')
parser.add_argument('--split', action='store_true', help='Split pairs to independent genes - no overlapping genes between test and train samples')
parser.add_argument('--semi', action='store_true', help='Semi supervised learning')
parser.add_argument('--random', action='store_true', help='Random features')
parser.add_argument('--compare', action='store_true', help='Compare to other model')
parser.add_argument('--rf_est', type=int, default=100, help='number of estimators for RF comparison')
parser.add_argument('--no_sp', action='store_true', help='Add cl2cl and drug2drug values')
args = parser.parse_args()


def gae_for_gi(args):
    final_res_auc, final_res_ap, final_res_auc_rf, final_res_ap_rf = np.zeros(args.nos), np.zeros(args.nos), np.zeros(args.nos), np.zeros(args.nos)
    best_model_path = join('Output', f'best-model_{args.dataset}_{args.num}.pt')
    plot_learning_file = join('Output', f'Loss_{args.dataset}_{args.num}.pt')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using {} dataset".format(args.dataset))
    adj, features, neutrals = load_data_gi(args.dataset, weighted=args.weighted, random_features=args.random)
    n_nodes, feat_dim = features.shape
    print(f"Num of nodes: {n_nodes} ; Features dim: {feat_dim}")

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    edges, edges_all, all_edge_idx_shuffled = prepare_for_cv(adj, weighted=args.weighted)
    if args.weighted or len(neutrals) > 0:
        if not args.all_neu:
            random_indices = np.random.choice(neutrals.shape[0], size=len(edges), replace=False)
            neutrals = neutrals[random_indices, :]
        all_neu_idx_shuffled = list(range(neutrals.shape[0]))
        np.random.shuffle(all_neu_idx_shuffled)
    if args.compare:
        with open("splits_compare.pkl", 'rb') as fp:
            dic_ind = pickle.load(fp)
    for i in range(args.nos):
        if args.compare:
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_compare(edges,
                                                                                                                                                neutrals, i, adj_orig.shape, dic_ind,
                                                                                                                                                all_neu=args.all_neu, no_sp=args.no_sp)
        elif args.weighted or len(neutrals) > 0:
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_gi_cv(edges, edges_all, all_edge_idx_shuffled, neutrals, all_neu_idx_shuffled, adj_orig.shape, i, total_splits=args.nos)
        else:
            train_edges_false=[]
            if args.split:
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_separate_genes(edges, edges_all, adj_orig.shape)
            else:
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_cv(edges, edges_all, all_edge_idx_shuffled, adj_orig.shape, i, total_splits=args.nos, semi=args.semi)
        print(f"train_edged: {len(train_edges)},{len(train_edges_false)}; val_edged: {len(val_edges)},{len(val_edges_false)}; test_edged: {len(test_edges)},{len(test_edges_false)}")

        if args.model == "vae2":
            model = GCNModelVAE2(feat_dim, args.hidden3, args.hidden1, args.hidden2, args.dropout)
        elif args.model == 'vaen':
            model = GCNModelVAENorm(feat_dim, args.hidden0, args.hidden1, args.hidden2, args.dropout)
        else:
            model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)

        adj_norm = train_model(adj_train, model, features, val_edges, val_edges_false, best_model_path, plot_learning_file, args.epochs, args.patience, args.lr, args.alpha, args.plt)
        final_res_auc[i], final_res_ap[i] = evaluate_model(model, adj_norm, features, test_edges, test_edges_false, best_model_path)
        final_res_auc_rf[i], final_res_ap_rf[i] = compare_to_rf_onto(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, [], np.array(features))

    print_results(final_res_auc, final_res_ap, final_res_auc_rf, final_res_ap_rf)



if __name__ == '__main__':
    gae_for_gi(args)
