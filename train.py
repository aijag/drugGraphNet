from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import optim

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, load_data_gi, mask_test_edges_gi, \
    mask_test_edges_cv, prepare_for_cv, mask_test_edges_gi_cv, mask_test_edges_cv_temp, create_double_features, \
    mask_test_edges_separate_genes, mask_test_edges_cv_temp_separated
from evaluation import get_roc_score
from vgnae_model import GAE, VGAE, Encoder

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden0', type=int, default=128, help='Number of units in gnae layer.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--weighted', type=bool, default=False, help='weight')
parser.add_argument('--num', type=int, default=0, help='number of experiment')
parser.add_argument('--nos', type=int, default=5, help='Number of splits for CV')
parser.add_argument('--sep', action='store_true', help='Split pairs to independent genes - no overlapping genes between test and train samples')
parser.add_argument('--rg', action='store_true', help='Predict regression')  # TODO: add regression
parser.add_argument('--semi', action='store_true', help='Semi supervised learning')

args = parser.parse_args()

def compare_to_rf_onto(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, features):
    train_pairs, train_labels, test_pairs, test_labels = mask_test_edges_cv_temp(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false)
    print("1")
    train_pairs = train_pairs.astype(int)
    test_pairs = test_pairs.astype(int)
    train_onto = create_double_features(train_pairs, np.array(features))
    test_onto = create_double_features(test_pairs, np.array(features))
    print("2")
    classifier = RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_split=5, n_jobs=-1)
    classifier.fit(train_onto, train_labels)
    print("3")
    pre = classifier.predict_proba(test_onto)[:,1]
    return roc_auc_score(test_labels, pre), average_precision_score(test_labels, pre)


def compare_to_rf_onto_split_genes(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, features):
    train_pairs, train_labels, test_pairs, test_labels = mask_test_edges_cv_temp_separated(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false)
    print("1")
    train_pairs = train_pairs.astype(int)
    test_pairs = test_pairs.astype(int)
    train_onto = create_double_features(train_pairs, np.array(features))
    test_onto = create_double_features(test_pairs, np.array(features))
    print("2")
    classifier = RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_split=5, n_jobs=-1)
    classifier.fit(train_onto, train_labels)
    print("3")
    pre = classifier.predict_proba(test_onto)[:,1]
    return roc_auc_score(test_labels, pre), average_precision_score(test_labels, pre)


def gae_for_gi(args):
    final_res_auc = []
    final_res_ap = []
    final_res_auc_rf = []
    final_res_ap_rf = []
    best_model_path = f'best-model_{args.dataset}_{args.num}.pt'
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using {} dataset".format(args.dataset))
    if args.dataset =='cora':
        adj, features = load_data(args.dataset)
    else:
        adj, features, neutrals = load_data_gi(args.dataset, weighted=args.weighted, random_features=False)
    n_nodes, feat_dim = features.shape
    print(f"Num of nodes: {n_nodes} ; Features dim: {feat_dim}")

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    edges, edges_all, all_edge_idx_shuffled = prepare_for_cv(adj, weighted=args.weighted)
    if args.weighted:
        all_neu_idx_shuffled = list(range(neutrals.shape[0]))
        np.random.shuffle(all_neu_idx_shuffled)
    for i in range(args.nos):
        if args.weighted:
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_gi_cv(edges, edges_all, all_edge_idx_shuffled, neutrals, all_neu_idx_shuffled, adj_orig.shape, i, total_splits=args.nos)
        else:
            if args.sep:
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_separate_genes(edges, edges_all, adj_orig.shape)
            else:
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_cv(edges, edges_all, all_edge_idx_shuffled, adj_orig.shape, i, total_splits=args.nos, semi=args.semi)
        adj = adj_train

        # Some preprocessing
        adj_norm = preprocess_graph(adj)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        if args.model=='gnae':
            model = GAE(Encoder(feat_dim, args.hidden0, train_edges, 'GAE')).to(dev)
        elif args.model=='vgnae':
            model = VGAE(Encoder(feat_dim, args.hidden0, train_edges, 'VGAE')).to(dev)
        else:
            model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        hidden_emb = None
        best_val_auc = 0
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            if args.model=='gnae' or args.model=='vgnae':
                train_edges_t = torch.LongTensor(train_edges.T)
                hidden_emb = model.encode(features, train_edges_t)
                loss = model.recon_loss(hidden_emb, train_edges_t)
                if args.model == 'vgnae':
                    loss = loss + (1 / n_nodes) * model.kl_loss()
                loss.backward()
                cur_loss = loss.item()
                optimizer.step()
                with torch.no_grad():
                    z = model.encode(features, train_edges_t)
                roc_curr, ap_curr = model.test(z, torch.LongTensor(val_edges.T), torch.LongTensor(val_edges_false.T))
            else:
                recovered, mu, logvar = model(features, adj_norm)
                loss = loss_function(preds=recovered, labels=adj_label,
                                     mu=mu, logvar=logvar, n_nodes=n_nodes,
                                     norm=norm, pos_weight=pos_weight)
                loss.backward()
                cur_loss = loss.item()
                optimizer.step()
                hidden_emb = mu.data.numpy()
                roc_curr, ap_curr = get_roc_score(hidden_emb, val_edges, val_edges_false)

            if (epoch+1) % 10 == 0:
                print(f"Epoch: {epoch + 1:04d} train_loss= {cur_loss:.5f} val_roc= {roc_curr:.5f} val_ap= {ap_curr:.5f} time= {time.time() - t:.5f}")
            if roc_curr > best_val_auc:
                torch.save(model.state_dict(), best_model_path)
                best_val_auc = roc_curr

        print("Optimization Finished!")
        model.load_state_dict(torch.load(best_model_path))
        if args.model == 'gnae' or args.model == 'vgnae':
            with torch.no_grad():
                z = model.encode(features, train_edges_t)
            roc_score, ap_score = model.test(z, torch.LongTensor(test_edges.T), torch.LongTensor(test_edges_false.T))
        else:
            recovered, mu, logvar = model(features, adj_norm)
            hidden_emb = mu.data.numpy()
            roc_score, ap_score = get_roc_score(hidden_emb, test_edges, test_edges_false)
        print(f'Test ROC score: {roc_score:.5}')
        print(f'Test AP score: {ap_score:.5}')
        final_res_auc.append(roc_score)
        final_res_ap.append(ap_score)
        if args.sep:
            auc2, ap2 = compare_to_rf_onto_split_genes(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, np.array(features))
        else:
            auc2, ap2 = compare_to_rf_onto(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, np.array(features))
        print(f'AUC2: {auc2:.5}')
        final_res_auc_rf.append(auc2)
        print(f'AP2: {ap2:.5}')
        final_res_ap_rf.append(ap2)
    print(final_res_auc)
    print(f'AUC: {np.average(final_res_auc):.5}')
    print(final_res_ap)
    print(f'AP: {np.average(final_res_ap):.5}')
    print(final_res_auc_rf)
    print(f'AUC-RF: {np.average(final_res_auc_rf):.5}')
    print(final_res_ap_rf)
    print(f'AP-RF: {np.average(final_res_ap_rf):.5}')


if __name__ == '__main__':
    gae_for_gi(args)
