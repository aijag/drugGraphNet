from os.path import join
import argparse

import numpy as np
import torch

from load_data import load_data_drug
from model import GCNModelVAE, GCNModelVAE2, GCNModelVAENorm

from train_model import train_model, evaluate_model
from utils import prepare_for_cv_score, mask_test_edges_drug, prepare_for_cv_drug_gi, mask_test_edges_drug_gi, compare_to_rf_onto, print_results

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vgaen', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--patience', type=int, default=10, help='Patience before early stopping of the training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden0', type=int, default=128, help='Number of units in gnae layer.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 3.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.5, help='vae proportion loss')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='data_drug', help='type of dataset.')
parser.add_argument('--net', type=str, default='GI', help='gene-gene type of connections, supports GI or PPI')
parser.add_argument('--num', type=int, default=0, help='number of experiment')
parser.add_argument('--all_net', action='store_true', help='Include all PPI/GI original net')
parser.add_argument('--all_neu', action='store_true', help='Include all neutral samples - data is unbalanced')
parser.add_argument('--random_f', action='store_true', help='Random features')
parser.add_argument('--random_s', action='store_true', help='Random structure')
parser.add_argument('--bin', action='store_true', help='Binarized edges of adj matrix - ppi/gi net')
parser.add_argument('--semi', action='store_true', help='Semi supervised learning')
parser.add_argument('--p_gi', action='store_true', help='Predict GI instead of Drug sensitivity')
parser.add_argument('--split', action='store_true', help='split to independent cell lines-drugs or genes in GI')
parser.add_argument('--no_sp', action='store_true', help='Add cl2cl and drug2drug values')
parser.add_argument('--plt', action='store_true', help='Plot graphs')
parser.add_argument('--nos', type=int, default=5, help='Number of splits for CV')
# parser.add_argument('--loo', action='store_true', help='Leave one out - check')


def gae_for_drug(args):
    final_res_auc, final_res_ap, final_res_auc_rf, final_res_ap_rf = np.zeros(args.nos), np.zeros(args.nos), np.zeros(args.nos), np.zeros(args.nos)
    best_model_path = join('Output', f'best-model_{args.dataset}_{args.num}.pt')
    plot_learning_file = join('Output', f'Loss_{args.dataset}_{args.num}.pt')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using {args.dataset} dataset\nUsing {args.net} network")
    cell2gene, drug2gene, gi_adj, cell_drug_data, features = load_data_drug(args.dataset, args.net, args.all_net, random_features=args.random_f, random_structure=args.random_s)
    n_nodes, feat_dim = features.shape
    print(f"Num of nodes: {n_nodes} ; Features dim: {feat_dim}")

    if args.p_gi:
        edges_pos, edges_neu, edge_idx_shuffled, edge_idx_neu_shuffled, adj_drug = prepare_for_cv_drug_gi(cell_drug_data, gi_adj, size=cell2gene.shape[0]+drug2gene.shape[0])
    else:
        edges_pos, edges_neu, edge_idx_shuffled, edge_idx_neu_shuffled = prepare_for_cv_score(cell_drug_data)

    for i in range(args.nos):
        print(f"Split number {i}")
        if args.p_gi:
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_drug_gi(
                cell2gene, drug2gene,
                adj_drug, edges_pos,
                edges_neu,
                edge_idx_shuffled, edge_idx_neu_shuffled,
                i,
                total_splits=args.nos, no_sp=args.no_sp, split=args.split)
        else:
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_drug(cell2gene, drug2gene,
                                                                                                                                       gi_adj, edges_pos,
                                                                                                                                       edges_neu,
                                                                                                                                       edge_idx_shuffled,
                                                                                                                                       edge_idx_neu_shuffled,
                                                                                                                                       i,
                                                                                                                                       total_splits=args.nos, all_neu=args.all_neu, binarize=args.bin, semi=args.semi, no_sp=args.no_sp)

        if args.model == "vae2":
            model = GCNModelVAE2(feat_dim, args.hidden3, args.hidden1, args.hidden2, args.dropout)
        elif args.model == 'vaen':
            model = GCNModelVAENorm(feat_dim, args.hidden0, args.hidden1, args.hidden2, args.dropout)
        else:
            model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        adj_norm = train_model(adj_train, model, features, val_edges, val_edges_false, best_model_path, plot_learning_file, args.epochs, args.patience, args.lr, args.alpha, args.plt)

        final_res_auc[i], final_res_ap[i] = evaluate_model(model, adj_norm, features,  test_edges, test_edges_false, best_model_path)
        final_res_auc_rf[i], final_res_ap_rf[i] = compare_to_rf_onto(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, [], np.array(features))
    print_results(final_res_auc, final_res_ap, final_res_auc_rf, final_res_ap_rf)


if __name__ == '__main__':
    args = parser.parse_args()
    gae_for_drug(args)

