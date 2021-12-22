import os
from os.path import join
import argparse
import pickle
import time
import networkx as nx

import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import optim

from model import GCNModelVAE, GCNModelVAE2
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, load_data_gi, mask_test_edges_gi, \
    mask_test_edges_cv, prepare_for_cv, mask_test_edges_gi_cv, mask_test_edges_cv_temp, create_double_features, \
    mask_test_edges_separate_genes, mask_test_edges_cv_temp_separated, prepare_for_cv_drug, mask_test_edges_drug

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='data_drug', help='type of dataset.')
parser.add_argument('--net', type=str, default='GI', help='type of dataset.')
parser.add_argument('--weighted', action='store_true', help='weight')
parser.add_argument('--num', type=int, default=0, help='number of experiment')
parser.add_argument('--all_ppi', action='store_true', help='Include all PPI/GI original net')
parser.add_argument('--all_neu', action='store_true', help='Include all neutral samples - data is unbalanced')
parser.add_argument('--random', action='store_true', help='Random features')
parser.add_argument('--bin', action='store_true', help='Binarized edges of adj matrix - ppi/gi net')
parser.add_argument('--semi', action='store_true', help='Semi supervised learning')

args = parser.parse_args()


def create_mat(line_ind, ppi_ind, dic):
    line2gene = np.zeros([len(line_ind), len(ppi_ind)])
    for i, cell in enumerate(line_ind):
        for g in dic[cell]:
            try:
                line2gene[i][ppi_ind.index(g)] = 1
            except ValueError:
                continue
    return line2gene


def load_data_drug(data='data_drug', ppi_file_name="ppi_anat_m.edgelist", index_file="cell_drug_genes_ind_all", features_npz_file="sparse_onto_all.npz", random_features=False):
    data_path = join(os.getcwd(), data)
    ppi_path = join(data_path, ppi_file_name)
    scores_file_name = 'IC50_bin_list.csv'
    # start_ppi = 1208
    ## Read ppi_net
    ppi_net = nx.read_edgelist(ppi_path, nodetype=str, data=(('weight', float),))
    ## Read cell_drug main data
    cell_drug_data = pd.read_csv(join(data_path, scores_file_name), usecols=['line', 'drug', 'score'])
    ## Load features
    with open(join(data_path, "go_terms"), "rb") as fp:
        go = pickle.load(fp)
    with open(join(data_path, index_file), "rb") as fp:
        index = pickle.load(fp)
    cell_ind = index[:len(np.unique(cell_drug_data['line']))]
    drug_ind = index[len(cell_ind):len(cell_ind) + len(np.unique(cell_drug_data['drug']))]
    features_data = sp.load_npz(join(data_path, features_npz_file)).todense()
    features = pd.DataFrame(features_data, index=index, columns=go)
    ppi_nodes = list(ppi_net.nodes)
    prob_ind = [x for x in index[len(cell_ind)+len(drug_ind):] if x not in ppi_nodes]
    features = features.drop(prob_ind)
    ppi_ind = list(features.index[len(cell_ind) + len(np.unique(cell_drug_data['drug'])):])
    features = features.values
    ## load cell2gene and drug2gene
    with open(os.path.join(data_path, 'mutation_lines_dic.pkl'), 'rb') as fp:
        line_target_dic = pickle.load(fp)
    with open(os.path.join(data_path, 'drug_target_dic.pkl'), 'rb') as fp:
        drug_target_dic = pickle.load(fp)
    cell2gene = create_mat(cell_ind, ppi_ind, line_target_dic)
    drug2gene = create_mat(drug_ind, ppi_ind, drug_target_dic)
    ppi_adj = nx.adjacency_matrix(ppi_net, nodelist=ppi_ind)
    ## Change data to num of location
    #######################################
    dict = {g: i for i, g in enumerate(index[:len(cell_ind) + len(drug_ind)])}
    cell_drug_data.replace({"line": dict, "drug": dict}, inplace=True)
    if random_features:
        features = np.random.rand(features.shape[0], 256)
    features_torch = torch.FloatTensor(features)
    return cell2gene, drug2gene, ppi_adj, cell_drug_data, features_torch


def compare_to_rf_onto(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, features):
    train_pairs = np.vstack([train_edges, val_edges, train_edges_false, val_edges_false])
    train_labels = np.hstack([np.ones(len(train_edges)+len(val_edges)), np.zeros(len(train_edges_false)+len(val_edges_false))])
    test_pairs = np.vstack([test_edges, test_edges_false])
    test_labels = np.hstack([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])
    print("1")
    train_onto = create_double_features(train_pairs, np.array(features))
    test_onto = create_double_features(test_pairs, np.array(features))
    print("2")
    classifier = RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_split=5, n_jobs=-1)
    classifier.fit(train_onto, train_labels)
    print("3")
    pre = classifier.predict_proba(test_onto)[:,1]
    return roc_auc_score(test_labels, pre), average_precision_score(test_labels, pre)


def gae_for_drug(args):
    number_of_splits = 5
    patience = 100
    final_res_auc = []
    final_res_ap = []
    final_res_auc_rf = []
    final_res_ap_rf = []
    best_model_path = f'best-model_{args.dataset}_{args.num}.pt'

    print(f"Using {args.dataset} dataset")
    print(f"Using {args.net} network")
    # adj, features = load_data(args.dataset)
    # adj, features, neutrals, num_of_clines, num_of_drugs = load_data_drug(args.dataset)
    if args.net == "GI":
        if args.all_ppi:
            # cell2gene, drug2gene, ppi_adj, cell_drug_data, features = load_data_drug(args.dataset, ppi_file_name="gi_sl_all.edgelist", index_file="gi_index", features_npz_file="sparse_onto_gi.npz")
            cell2gene, drug2gene, ppi_adj, cell_drug_data, features = load_data_drug(args.dataset, ppi_file_name="gi_sl2.edgelist", index_file="gi_index2", features_npz_file="sparse_onto_gi2.npz")
        else:
            cell2gene, drug2gene, ppi_adj, cell_drug_data, features = load_data_drug(args.dataset, ppi_file_name="gi_sl.edgelist", index_file="gi_index", features_npz_file="sparse_onto_gi.npz")
    elif args.net == "PPI":
        if args.all_ppi:
            cell2gene, drug2gene, ppi_adj, cell_drug_data, features = load_data_drug(args.dataset, ppi_file_name="ppi_anat_m_all.edgelist")
        else:
            cell2gene, drug2gene, ppi_adj, cell_drug_data, features = load_data_drug(args.dataset, random_features=args.random)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    # adj_orig = adj
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()
    # edges, edges_all, all_edge_idx_shuffled = prepare_for_cv(adj, weighted=args.weighted)
    edges_pos, edges_neu, edge_idx_shuffled, edge_idx_neu_shuffled = prepare_for_cv_drug(cell_drug_data)
    # if args.weighted:
    #     all_neu_idx_shuffled = list(range(neutrals.shape[0]))
    #     np.random.shuffle(all_neu_idx_shuffled)
    for i in range(number_of_splits):
        print(f"Split number {i}")
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_drug(cell2gene, drug2gene,
                                                                                                   ppi_adj, edges_pos,
                                                                                                   edges_neu,
                                                                                                   edge_idx_shuffled,
                                                                                                   edge_idx_neu_shuffled,
                                                                                                   i,
                                                                                                   total_splits=number_of_splits, all_neu=args.all_neu, binarize=args.bin, semi=args.semi)
        adj = adj_train
        # Some preprocessing
        adj_norm = preprocess_graph(adj)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        if args.model == "gcn_vae2":
            model = GCNModelVAE2(feat_dim, args.hidden3, args.hidden1, args.hidden2, args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        hidden_emb = None
        best_val_auc = 0
        trigger_times = 0
        the_last_roc = 0
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
            roc_curr, ap_curr = get_roc_score(hidden_emb, val_edges, val_edges_false)
            if (epoch+1) % 20 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "val_roc=", "{:.5f}".format(roc_curr),
                      "val_ap=", "{:.5f}".format(ap_curr),
                      "time=", "{:.5f}".format(time.time() - t)
                      )
            if roc_curr > best_val_auc:
                torch.save(model.state_dict(), best_model_path)
                best_val_auc = roc_curr

            # Early stopping
            if roc_curr < the_last_roc:
                trigger_times += 1
                if trigger_times%10==0:
                    print('trigger times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart the test process.')
                    break
            else:
                trigger_times = 0
            the_last_roc = roc_curr

        print("Optimization Finished!")
        model.load_state_dict(torch.load(best_model_path))
        recovered, mu, logvar = model(features, adj_norm)
        hidden_emb = mu.data.numpy()
        roc_score, ap_score = get_roc_score(hidden_emb, test_edges, test_edges_false)
        print(f'Test ROC score: {roc_score:.5}')
        print(f'Test AP score: {ap_score:.5}')
        final_res_auc.append(roc_score)
        final_res_ap.append(ap_score)
        auc2, ap2 = compare_to_rf_onto(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, np.array(features))
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
    gae_for_drug(args)
