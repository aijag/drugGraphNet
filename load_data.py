import os
import pickle
import pickle as pkl
import sys

from itertools import combinations
from random import sample
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from os.path import join
import pandas as pd
import random
from constants import *


### Utils function for loading the data ###
def create_mat(line_ind, ppi_ind, dic):
    line2gene = np.zeros([len(line_ind), len(ppi_ind)])
    for i, cell in enumerate(line_ind):
        for g in dic[cell]:
            if g in ppi_ind:
                line2gene[i][ppi_ind.index(g)] = 1
    return line2gene

def shuffle_dic(dic):
    keys = list(dic.keys())
    random.shuffle(keys)
    return {keys[i]: val for i, val in enumerate(dic.values())}


def get_dep_drug_data(data_path, net, all_net):
    if net == "GI":
        if all_net:
            net_file_name, index_file, features_npz_file = gi_net_file, gi_index_file, gi_features_file
        else:
            net_file_name, index_file, features_npz_file = gi_net_p_file, gi_index_file, gi_features_file
    elif net == "PPI":
        if all_net:
            net_file_name, index_file, features_npz_file = ppi_net_file, ppi_index_file, ppi_features_file
        else:
            net_file_name, index_file, features_npz_file = ppi_net_p_file, ppi_index_file, ppi_features_file
    else:
        print("Error: Unfamiliar network is requested.")
        sys.exit()
    net_path = join(data_path, net_file_name)
    ## Read gi/ppi net
    genes_net = nx.read_edgelist(net_path, nodetype=str, data=(('weight', float),))
    ## Load features
    with open(join(data_path, go_terms_file), "rb") as fp:
        go = pickle.load(fp)
    with open(join(data_path, index_file), "rb") as fp:
        index = pickle.load(fp)
    features_data = sp.load_npz(join(data_path, features_npz_file)).toarray()
    features = pd.DataFrame(features_data, index=index, columns=go)
    return features, index, genes_net


#### Loading main data ####

### Load GI ###
def load_data_gi(data, weighted=False, random_features=False):
    # edge_file = "neg.edgelist", onto_file = "onto_genes.csv"
    data_path = join(os.getcwd(), "Data", data)
    features_path = join(data_path, features_file)
    features = pd.read_csv(features_path, index_col=0)
    if weighted or data == "data_sl":
        if weighted:
            graph_path = join(data_path, edges_file_weighted)
            G = nx.read_edgelist(graph_path, nodetype=str, data=(('weight', int),))
        else:
            graph_path = join(data_path, edges_file_neg)
            G = nx.read_edgelist(graph_path, nodetype=str)
        edges = nx.to_pandas_edgelist(G).set_index(['source', 'target'])
        if weighted:
            nodes_names = features.index.to_list()
            adj = nx.adjacency_matrix(G, nodes_names, weight="weight")
            neutrals = edges[edges.weight == 0].index.to_list()
            neutrals = np.array([[nodes_names.index(pair[0]), nodes_names.index(pair[1])] for pair in neutrals])
        else:
            features_genes = list(features.index)
            t = [x for x in features_genes if x not in G.nodes]
            features = features.drop(t)
            nodes_names = features.index.to_list()
            adj = nx.adjacency_matrix(G, nodes_names)
            neutrals = []
    else:
        dataset = pd.read_csv(join(data_path, data_file_name))
        dict = {g: i for i, g in enumerate(features.index.to_list())}
        if data == "data_drug":
            dataset['line'] = dataset['line'].map(dict.get)
            dataset['drug'] = dataset['drug'].map(dict.get)
        elif data == "data_dependency" or data == "data_dependency19":
            dataset['line'] = dataset['line'].map(dict.get)
            dataset['gene'] = dataset['gene'].map(dict.get)
        else:
            dataset['gene1'] = dataset['gene1'].map(dict.get)
            dataset['gene2'] = dataset['gene2'].map(dict.get)
        neutrals = dataset[dataset.score == 0].drop('score', axis=1).values
        positives = dataset[dataset.score == 1]
        data_t = np.ones(positives.shape[0])
        adj = sp.csr_matrix((data_t, (positives.values[:, 0], positives.values[:, 1])),
                                    shape=[features.shape[0], features.shape[0]])
        adj = adj + adj.T  # turn the matrix to symmetric, since the original adj in drug_data isn't symmetric(based on pairs and not edgelist)
    if random_features:
        # identity_features = np.identity(features.shape[0]).astype(int)
        features = np.random.rand(features.shape[0], features.shape[1])
        features_torch = torch.FloatTensor(features)
    else:
        features_torch = torch.FloatTensor(features.values)
    return adj, features_torch, neutrals


def load_data_dependency(data, net, all_net, random_features=False, random_structure=False):
    data_path = join(os.getcwd(), "Data", data)
    features, index, genes_net = get_dep_drug_data(data_path, net, all_net)

    ## Read cell_drug main data
    cell_gene_data = pd.read_csv(join(data_path, data_file_name), usecols=['line', 'gene', 'score'])
    cell_ind = index[:len(np.unique(cell_gene_data['line']))]
    genes_nodes = list(genes_net.nodes)
    prob_ind = [x for x in index[len(cell_ind):] if x not in genes_nodes]
    features = features.drop(prob_ind)
    gene_ind = index[len(cell_ind):]
    features = features.values
    ## Change data to num of location
    #######################################
    dict = {g: i for i, g in enumerate(index[:len(cell_ind) + len(gene_ind)])}
    cell_gene_data['line'] = cell_gene_data['line'].map(dict.get)
    cell_gene_data['gene'] = cell_gene_data['gene'].map(dict.get)
    genes_edges = [[dict[g[0]], dict[g[1]]] for g in genes_net.edges]
    if random_structure:
        genes_values = np.unique(cell_gene_data['gene'])
        genes_edges = sample(list(combinations(genes_values, 2)), len(genes_edges))
    if random_features:
        features = np.random.rand(features.shape[0], 256)
    features_torch = torch.FloatTensor(features)
    return cell_gene_data, genes_edges, features_torch


### Data drug main data ###
def load_data_drug(data, net, all_net, random_features=False, random_structure=False):
    data_path = join(os.getcwd(), "Data", data)
    features, index, genes_net = get_dep_drug_data(data_path, net, all_net)

    ## Read cell_drug main data
    cell_drug_data = pd.read_csv(join(data_path, data_file_name), usecols=['line', 'gene', 'score'])
    cell_ind = index[:len(np.unique(cell_drug_data['line']))]
    drug_ind = index[len(cell_ind):len(cell_ind) + len(np.unique(cell_drug_data['drug']))]
    genes_nodes = list(genes_net.nodes)
    prob_ind = [x for x in index[len(cell_ind) + len(drug_ind):] if x not in genes_nodes]
    features = features.drop(prob_ind)
    gene_ind = list(features.index[len(cell_ind) + len(np.unique(cell_drug_data['drug'])):])
    features = features.values
    ## load cell2gene and drug2gene
    with open(os.path.join(data_path, mutation_dic), 'rb') as fp:
        line_target_dic = pickle.load(fp)
    with open(os.path.join(data_path, drug_targets_dic), 'rb') as fp:
        drug_target_dic = pickle.load(fp)
    if random_structure:
        line_target_dic = shuffle_dic(line_target_dic)
        drug_target_dic = shuffle_dic(drug_target_dic)
    cell2gene = create_mat(cell_ind, gene_ind, line_target_dic)
    drug2gene = create_mat(drug_ind, gene_ind, drug_target_dic)
    if random_structure:
        random.shuffle(gene_ind)
    genes_adj = nx.adjacency_matrix(genes_net, nodelist=gene_ind)
    ## Change data to num of location
    #######################################
    dict = {g: i for i, g in enumerate(index[:len(cell_ind) + len(drug_ind)])}
    cell_drug_data['line'] = cell_drug_data['line'].map(dict.get)
    cell_drug_data['drug'] = cell_drug_data['drug'].map(dict.get)
    if random_features:
        features = np.random.rand(features.shape[0], 256)
    features_torch = torch.FloatTensor(features)
    return cell2gene, drug2gene, genes_adj, cell_drug_data, features_torch
