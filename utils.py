import os
import numpy as np
import scipy.sparse as sp
import torch
from os.path import join
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier


def print_results(final_res_auc, final_res_ap, final_res_auc_rf, final_res_ap_rf):
    print(final_res_auc)
    print(f'AUC: {np.average(final_res_auc):.5}')
    print(final_res_ap)
    print(f'AP: {np.average(final_res_ap):.5}')
    if final_res_auc_rf:
        print(final_res_auc_rf)
        print(f'AUC-RF: {np.average(final_res_auc_rf):.5}')
        print(final_res_ap_rf)
        print(f'AP-RF: {np.average(final_res_ap_rf):.5}')


def create_ppi_net(ppi_net, ppi_ind):
    gene2gene = np.zeros([len(ppi_ind), len(ppi_ind)])
    for pair in ppi_net:
        ind1 = ppi_ind.index(pair[0])
        ind2 = ppi_ind.index(pair[1])
        gene2gene[ind1][ind2] = 1
        gene2gene[ind2][ind1] = 1
    return gene2gene


def rebuild_adj(train_edges, adj_shape):
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj_shape)
    adj_train = adj_train + adj_train.T
    return adj_train


def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)


def sparse_to_tuple(sparse_mx, weighted=False):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    if weighted:
        coords = coords[values == 1]
    return coords, values, shape


def get_edges(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    adj_triu = sp.triu(adj)
    return sparse_to_tuple(adj_triu)[0]


def prepare_for_cv(adj, weighted=False):
    edges = get_edges(adj)
    edges_all = sparse_to_tuple(adj, weighted=weighted)[0]
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    return edges, edges_all, all_edge_idx


def prepare_for_cv_score(data, column_score_name='score'):
    edges_pos = data[data[column_score_name] == 1].drop(columns=column_score_name).values
    edges_neu = data[data[column_score_name] == 0].drop(columns=column_score_name).values
    edge_idx = list(range(edges_pos.shape[0]))
    np.random.shuffle(edge_idx)
    edge_idx_neu = list(range(edges_neu.shape[0]))
    np.random.shuffle(edge_idx_neu)
    return edges_pos, edges_neu, edge_idx, edge_idx_neu

def prepare_for_cv_dep_gi(edges, dep_data, column_score_name='score'):
    values = set(np.unique(edges))
    neutrals = []
    samples = 0
    while samples < len(edges):
        idx_i, idx_j = random.sample(values, 2)
        if ismember([idx_i, idx_j], edges):
            continue
        if ismember([idx_j, idx_i], edges):
            continue
        if samples > 0:
            if ismember([idx_j, idx_i], neutrals):
                continue
            if ismember([idx_i, idx_j], neutrals):
                continue
        neutrals.append([idx_i, idx_j])
        samples += 1
    edge_idx_shuffled = list(range(edges.shape[0]))
    np.random.shuffle(edge_idx_shuffled)
    edge_idx_neu_shuffled = list(range(neutrals.shape[0]))
    np.random.shuffle(edge_idx_neu_shuffled)
    dep_pairs = dep_data[dep_data[column_score_name] == 1].drop(columns=column_score_name).values
    return edges, neutrals, edge_idx_shuffled, edge_idx_neu_shuffled, dep_pairs

def prepare_for_cv_drug_gi(data, gi_adj, size):
    edges_drug = data[data.score == 1].drop(columns='score').values
    data_drug = np.ones(edges_drug.shape[0])
    adj_drug = sp.csr_matrix((data_drug, (edges_drug[:, 0], edges_drug[:, 1])), shape=[size, size])
    adj_drug = (adj_drug + adj_drug.T).toarray()
    gi_adj[gi_adj > 0] = 1
    edges, edges_all, all_edge_idx_shuffled = prepare_for_cv(gi_adj)
    edges_neu = create_false_edges(edges_all, len(edges), np.max(edges))
    edge_idx_neu_shuffled = list(range(edges.shape[0]))
    np.random.shuffle(edge_idx_neu_shuffled)
    return edges, edges_neu, all_edge_idx_shuffled, edge_idx_neu_shuffled, adj_drug


def mask_test_edges_cv(edges, edges_all, all_edge_idx, adj_shape, i, total_splits=6, semi=False):
    num_test = int(np.floor(edges.shape[0] / total_splits))
    num_val = int(np.floor(edges.shape[0] / (2*total_splits)))
    test_edge_idx = all_edge_idx[i*num_test: (i+1)*num_test]
    val_edge_idx = all_edge_idx[(i+1)*num_test:((i+1)*num_test+num_val)]
    if i == total_splits-1:
        val_edge_idx = all_edge_idx[0: num_val]
    val_edges = edges[val_edge_idx]
    if semi:
        train_edges = edges[test_edge_idx]
        test_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    else:
        test_edges = edges[test_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    test_edges_false = np.zeros([len(test_edges), 2])
    samples = 0
    while samples < len(test_edges):
        idx_i = np.random.randint(0, adj_shape[0])
        idx_j = np.random.randint(0, adj_shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if samples>0:
            if ismember([idx_j, idx_i], test_edges_false):
                continue
            if ismember([idx_i, idx_j], test_edges_false):
                continue
        test_edges_false[samples, :] = [idx_i, idx_j]
        samples += 1
    val_edges_false = np.zeros([len(val_edges), 2])
    samples = 0
    while samples < len(val_edges):
        idx_i = np.random.randint(0, adj_shape[0])
        idx_j = np.random.randint(0, adj_shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], test_edges_false):
            continue
        if ismember([idx_i, idx_j], test_edges_false):
            continue
        if samples > 0:
            if ismember([idx_j, idx_i], val_edges_false):
                continue
            if ismember([idx_i, idx_j], val_edges_false):
                continue
        val_edges_false[samples, :] = [idx_i, idx_j]
        samples += 1
    adj_train = rebuild_adj(train_edges, adj_shape)
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def pair_in_list(pair, list):
    return all(x in list for x in pair)

def mask_test_edges_separate_genes(edges, edges_all, adj_shape):
    all_genes = np.arange(adj_shape[0])
    np.random.shuffle(all_genes)
    num_test = int(np.floor(len(all_genes) / 4.))
    num_val = int(np.floor(len(all_genes) / 10.))
    test_genes = all_genes[:num_test]
    val_genes = all_genes[num_test: num_test+num_val]
    train_genes = all_genes[num_test+num_val:]
    test_edges = []
    val_edges = []
    train_edges = []
    for pair in edges:
        if pair_in_list(pair, test_genes):
            test_edges.append(pair)
        elif pair_in_list(pair, val_genes):
            val_edges.append(pair)
        elif pair_in_list(pair, train_genes):
            train_edges.append(pair)
    test_edges = np.array(test_edges)
    val_edges = np.array(val_edges)
    train_edges = np.array(train_edges)
    ###################################################
    ###################################################
    test_edges_false = np.zeros([len(test_edges), 2])
    samples = 0
    while samples < len(test_edges):
        idx_i, idx_j = random.choices(test_genes, k=2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if samples > 0:
            if ismember([idx_j, idx_i], test_edges_false):
                continue
            if ismember([idx_i, idx_j], test_edges_false):
                continue
        test_edges_false[samples, :] = [idx_i, idx_j]
        samples += 1
    val_edges_false = np.zeros([len(val_edges), 2])
    samples = 0
    while samples < len(val_edges):
        idx_i, idx_j = random.choices(val_genes, k=2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], test_edges_false):
            continue
        if ismember([idx_i, idx_j], test_edges_false):
            continue
        if samples > 0:
            if ismember([idx_j, idx_i], val_edges_false):
                continue
            if ismember([idx_i, idx_j], val_edges_false):
                continue
        val_edges_false[samples, :] = [idx_i, idx_j]
        samples += 1
    adj_train = rebuild_adj(train_edges, adj_shape)
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


## Second split version - we will include all the sumples this time - CV version
def mask_test_edges_gi_cv(edges, edges_all, all_edge_idx, neutrals, all_neu_idx, adj_shape, i, total_splits=6):
    num_test = int(np.floor(edges.shape[0] / total_splits))
    num_val = int(np.floor(edges.shape[0] / (2*total_splits)))
    test_edge_idx = all_edge_idx[i*num_test: (i+1)*num_test]
    val_edge_idx = all_edge_idx[(i+1)*num_test:((i+1)*num_test+num_val)]
    if i == total_splits-1:
        val_edge_idx = all_edge_idx[0: num_val]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    num_neg_test = int(np.floor(len(neutrals) / total_splits))
    num_neg_val = int(np.floor(len(neutrals) / (2*total_splits)))
    test_edge_idx = all_neu_idx[i*num_neg_test: (i+1)*num_neg_test]
    val_edge_idx = all_neu_idx[(i+1)*num_neg_test:((i+1)*num_neg_test+num_neg_val)]
    if i == total_splits-1:
        val_edge_idx = all_neu_idx[0: num_val]
    test_edges_false = neutrals[test_edge_idx]
    val_edges_false = neutrals[val_edge_idx]
    train_edges_false = np.delete(neutrals, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    adj_train = rebuild_adj(train_edges, adj_shape)
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def split_train_test_edges(edges, edges_neutrals, edge_idx, edge_idx_neu, i, total_splits, all_neu, binarize, semi):
    num_test = int(np.floor(edges.shape[0] / total_splits))
    num_val = int(np.floor(edges.shape[0] / (2 * total_splits)))
    test_edge_idx = edge_idx[i * num_test: (i + 1) * num_test]
    val_edge_idx = edge_idx[(i + 1) * num_test:((i + 1) * num_test + num_val)]
    if i == total_splits - 1:
        val_edge_idx = edge_idx[0: num_val]
    val_edges = edges[val_edge_idx]
    if semi:
        train_edges = edges[test_edge_idx]
        test_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    else:
        test_edges = edges[test_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    if all_neu:
        num_test = int(np.floor(edges_neutrals.shape[0] / total_splits))
        num_val = int(np.floor(edges_neutrals.shape[0] / (2 * total_splits)))
    test_edge_idx = edge_idx_neu[i * num_test: (i + 1) * num_test]
    val_edge_idx = edge_idx_neu[(i + 1) * num_test:((i + 1) * num_test + num_val)]
    if i == total_splits - 1:
        val_edge_idx = edge_idx_neu[0: num_val]
    val_edges_false = edges_neutrals[val_edge_idx]
    if semi:
        train_edges_false = edges_neutrals[test_edge_idx]
        test_edges_false = np.delete(edges_neutrals, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        if not all_neu:
            test_edges_false = test_edges_false[np.random.randint(test_edges_false.shape[0], size=len(test_edges)), :]
    else:
        test_edges_false = edges_neutrals[test_edge_idx]
        train_edges_false = np.delete(edges_neutrals, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        if not all_neu:
            train_edges_false = train_edges_false[np.random.randint(train_edges_false.shape[0], size=len(train_edges)),:]
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def split_genes_compare(edges, edges_neutrals, i, dic_ind):
    train_ind, val_ind, test_ind = dic_ind[i]['train'], dic_ind[i]['val'], dic_ind[i]['test']
    train_edges = np.array([pair for pair in edges if pair[0] in train_ind])
    val_edges = np.array([pair for pair in edges if pair[0] in val_ind])
    test_edges = np.array([pair for pair in edges if pair[0] in test_ind])
    train_edges_false = np.array([pair for pair in edges_neutrals if pair[0] in train_ind])
    val_edges_false = np.array([pair for pair in edges_neutrals if pair[0] in val_ind])
    test_edges_false = np.array([pair for pair in edges_neutrals if pair[0] in test_ind])
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_compare(edges, edges_neutrals, i, adj_shape, dic_ind, all_neu=False, no_sp=False, gi=[]):
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = split_genes_compare(edges, edges_neutrals, i, dic_ind)
    if gi:
        train_edges_t = train_edges + gi
    else:
        train_edges_t = train_edges
    data = np.ones(train_edges_t.shape[0])
    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges_t[:, 0], train_edges_t[:, 1])), shape=adj_shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_dep_cv(edges, edge_idx, edges_neutrals, edge_idx_neu, gi, adj_shape, i, total_splits=5, all_neu=False, binarize=False, semi=False):
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = split_train_test_edges(
        edges, edges_neutrals, edge_idx, edge_idx_neu, i, total_splits, all_neu, binarize, semi)
    if gi:
        train_edges_t = np.concatenate([train_edges, np.array(gi)])
    else:
        train_edges_t = train_edges
    adj_train = rebuild_adj(train_edges_t, adj_shape)
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_drug(cell2gene, drug2gene, ppi_adj, edges, edges_neutrals, edge_idx, edge_idx_neu, i, total_splits=5, all_neu=False, binarize=False, semi=False, no_sp=False):
    ## Create neutral and positive edges for train/val/test
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = split_train_test_edges(
        edges, edges_neutrals, edge_idx, edge_idx_neu, i, total_splits, all_neu, binarize, semi)
    adj_predict = rebuild_adj(train_edges, [cell2gene.shape[0]+drug2gene.shape[0],cell2gene.shape[0]+drug2gene.shape[0]])
    if no_sp:
        data_path = join(os.getcwd(), "data_drug")
        cl2cl = np.load(join(data_path, "cl2cl.npy"))
        drug2drug = np.load(join(data_path, "drug2drug.npy"))
        cl2cl = cl2cl - np.identity(cl2cl.shape[0])
        drug2drug = np.nan_to_num(drug2drug)
        drug2drug = drug2drug - np.identity(drug2drug.shape[0])
        adj_predict[:cl2cl.shape[0], :cl2cl.shape[0]] = cl2cl
        adj_predict[cl2cl.shape[0]:cl2cl.shape[0]+drug2drug.shape[0], cl2cl.shape[0]:cl2cl.shape[0]+drug2drug.shape[0]] = drug2drug
    temp1 = np.concatenate((cell2gene, drug2gene), axis=0)
    temp1 = np.concatenate((adj_predict, temp1), axis=1)
    adj_array = np.nan_to_num(ppi_adj.toarray())
    if binarize:
        adj_array = np.where(adj_array > 0, 1, 0)
    temp2 = np.concatenate((cell2gene.T, drug2gene.T, adj_array), axis=1) ### New change to remove the nans - need to recheck later
    adj = np.concatenate((temp1, temp2), axis=0)
    adj_train = sp.csr_matrix(adj)
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_drug_gi(cell2gene, drug2gene, adj_drug, edges, edges_neutrals, edge_idx, edge_idx_neu, i, total_splits=6, no_sp=False, split=False):
    ## Create neutral and positive edges for train/val/test
    if split:
        all_genes = np.arange(cell2gene.shape[1])
        np.random.shuffle(all_genes)
        num_test = int(np.floor(len(all_genes) / 4.))
        num_val = int(np.floor(len(all_genes) / 10.))
        test_genes = all_genes[:num_test]
        val_genes = all_genes[num_test: num_test + num_val]
        train_genes = all_genes[num_test + num_val:]
        test_edges, val_edges, train_edges = [], [], []
        for pair in edges:
            if pair_in_list(pair, test_genes):
                test_edges.append(pair)
            elif pair_in_list(pair, val_genes):
                val_edges.append(pair)
            elif pair_in_list(pair, train_genes):
                train_edges.append(pair)
        test_edges, val_edges, train_edges = np.array(test_edges), np.array(val_edges), np.array(train_edges)
        test_edges_false, val_edges_false, train_edges_false = [], [], []
        for pair in edges_neutrals:
            if pair_in_list(pair, test_genes):
                test_edges_false.append(pair)
            elif pair_in_list(pair, val_genes):
                val_edges_false.append(pair)
            elif pair_in_list(pair, train_genes):
                train_edges_false.append(pair)
        test_edges_false, val_edges_false, train_edges_false = np.array(test_edges_false), np.array(val_edges_false), np.array(train_edges_false)
    else:
        num_test = int(np.floor(edges.shape[0] / total_splits))
        num_val = int(np.floor(edges.shape[0] / (2 * total_splits)))
        test_edge_idx = edge_idx[i * num_test: (i + 1) * num_test]
        val_edge_idx = edge_idx[(i + 1) * num_test:((i + 1) * num_test + num_val)]
        if i == total_splits - 1:
            val_edge_idx = edge_idx[0: num_val]
        val_edges = edges[val_edge_idx]
        test_edges = edges[test_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        test_edge_idx = edge_idx_neu[i * num_test: (i + 1) * num_test]
        val_edge_idx = edge_idx_neu[(i + 1) * num_test:((i + 1) * num_test + num_val)]
        if i == total_splits - 1:
            val_edge_idx = edge_idx_neu[0: num_val]
        val_edges_false = edges_neutrals[val_edge_idx]
        test_edges_false = edges_neutrals[test_edge_idx]
        train_edges_false = np.delete(edges_neutrals, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    adj_predict = rebuild_adj(train_edges, [cell2gene.shape[1], cell2gene.shape[1]])
    if no_sp:
        data_path = join(os.getcwd(), "data_drug")
        cl2cl = np.load(join(data_path, "cl2cl.npy"))
        drug2drug = np.load(join(data_path, "drug2drug.npy"))
        cl2cl = cl2cl - np.identity(cl2cl.shape[0])
        drug2drug = np.nan_to_num(drug2drug)
        drug2drug = drug2drug - np.identity(drug2drug.shape[0])
        adj_drug[:cl2cl.shape[0], :cl2cl.shape[0]] = cl2cl
        adj_drug[cl2cl.shape[0]:cl2cl.shape[0] + drug2drug.shape[0],cl2cl.shape[0]:cl2cl.shape[0] + drug2drug.shape[0]] = drug2drug
    temp1 = np.concatenate((cell2gene, drug2gene), axis=0)
    temp1 = np.concatenate((adj_drug, temp1), axis=1)

    temp2 = np.concatenate((cell2gene.T, drug2gene.T, adj_predict), axis=1)
    adj = np.concatenate((temp1, temp2), axis=0)
    adj_train = sp.csr_matrix(adj)
    # Shift edges values by temp1.shape[0]
    train_edges += temp1.shape[0]
    train_edges_false += temp1.shape[0]
    val_edges += temp1.shape[0]
    val_edges_false += temp1.shape[0]
    test_edges += temp1.shape[0]
    test_edges_false += temp1.shape[0]
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_false_edges(edges_all, num_of_edges, max_num):
    edges_false = np.zeros([num_of_edges, 2])
    samples = 0
    while samples < edges_false.shape[0]:
        idx_i = np.random.randint(0, max_num)  # max_num=adj.shape[0]
        idx_j = np.random.randint(0, max_num)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if samples > 0:
            if ismember([idx_j, idx_i], edges_false):
                continue
            if ismember([idx_i, idx_j], edges_false):
                continue
        edges_false[samples, :] = [idx_i, idx_j]
        samples += 1
    return edges_false.astype(int)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


## Helper function for creating ontotypes or comparizon
def create_double_features(pairs, features, concatenate=False):
    if concatenate:
        final = np.zeros([len(pairs), 2*features.shape[1]])
        for i, pair in enumerate(pairs):
            final[i, :] = np.concatenate([features[pair[0], :],features[pair[1], :]])
    else:
        final = np.zeros([len(pairs), features.shape[1]])
        for i, pair in enumerate(pairs):
            final[i, :] = features[pair[0], :]+features[pair[1], :]
    return final

def compare_to_rf_onto(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, features,  n_estimators=100, concatenate=False, split_data=False):
    if split_data:
        train_pairs, train_labels, test_pairs, test_labels = mask_test_edges_cv_temp_separated(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false)
    elif len(train_edges_false) == 0:
        train_pairs, train_labels, test_pairs, test_labels = mask_test_edges_cv_temp(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false)
    else:
        train_pairs = np.vstack([train_edges, val_edges, train_edges_false, val_edges_false])
        train_labels = np.hstack([np.ones(len(train_edges) + len(val_edges)), np.zeros(len(train_edges_false) + len(val_edges_false))])
        test_pairs = np.vstack([test_edges, test_edges_false])
        test_labels = np.hstack([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])
    print("1")
    train_pairs = train_pairs.astype(int)
    test_pairs = test_pairs.astype(int)
    train_onto = create_double_features(train_pairs, np.array(features), concatenate=concatenate)
    test_onto = create_double_features(test_pairs, np.array(features), concatenate=concatenate)
    print("2")
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=100, min_samples_split=5, n_jobs=-1)
    classifier.fit(train_onto, train_labels)
    print("3")
    pre = classifier.predict_proba(test_onto)[:,1]
    auc, ap = roc_auc_score(test_labels, pre), average_precision_score(test_labels, pre)
    print(f'AUC_RF: {auc:.5} ; AP_RF: {ap:.5}')
    return auc, ap


## Temp split for including training data
def mask_test_edges_cv_temp(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false):
    all_genes = np.unique(edges_all)
    train_edges_false = np.zeros([len(train_edges), 2])
    samples = 0
    while samples < len(train_edges):
        idx_i, idx_j = random.choices(all_genes, k=2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], test_edges_false):
            continue
        if ismember([idx_i, idx_j], test_edges_false):
            continue
        if ismember([idx_j, idx_i], val_edges_false):
            continue
        if ismember([idx_i, idx_j], val_edges_false):
            continue
        if samples > 0:
            if ismember([idx_j, idx_i], train_edges_false):
                continue
            if ismember([idx_i, idx_j], train_edges_false):
                continue
        train_edges_false[samples, :] = [idx_i, idx_j]
        samples += 1

    train_pairs = np.vstack([train_edges, val_edges, train_edges_false, val_edges_false])
    train_labels = np.hstack(
        [np.ones(len(train_edges) + len(val_edges)), np.zeros(len(train_edges_false) + len(val_edges_false))])
    test_pairs = np.vstack([test_edges, test_edges_false])
    test_labels = np.hstack([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])
    return train_pairs, train_labels, test_pairs, test_labels


def mask_test_edges_cv_temp_separated(edges_all, train_edges, val_edges, val_edges_false, test_edges, test_edges_false):
    train_genes = np.unique(train_edges)
    train_edges_false = np.zeros([len(train_edges), 2])
    samples = 0
    while samples < len(train_edges):
        idx_i, idx_j = random.choices(train_genes, k=2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if samples > 0:
            if ismember([idx_j, idx_i], train_edges_false):
                continue
            if ismember([idx_i, idx_j], train_edges_false):
                continue
        train_edges_false[samples, :] = [idx_i, idx_j]
        samples += 1

    train_pairs = np.vstack([train_edges, val_edges, train_edges_false, val_edges_false])
    train_labels = np.hstack([np.ones(len(train_edges)+len(val_edges)), np.zeros(len(train_edges_false)+len(val_edges_false))])
    test_pairs = np.vstack([test_edges, test_edges_false])
    test_labels = np.hstack([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])
    return train_pairs, train_labels, test_pairs, test_labels
