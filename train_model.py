import time
import scipy.sparse as sp

import torch
from torch import optim

from evaluation import get_roc_score
from model import loss_function
from plots import plot_learning
from utils import preprocess_graph


def train_model(adj, model, features, val_edges, val_edges_false, best_model_path, plot_learning_file, epochs, patience=10, lr=0.01, alpha=0.5, plot_training=False):
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj + sp.eye(adj.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_auc, trigger_times, the_last_roc = 0, 0, 0
    train_loss, val_roc = [], []
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=features.shape[0],
                             norm=norm, pos_weight=pos_weight, alpha=alpha)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        # hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(mu.data.numpy(), val_edges, val_edges_false)
        train_loss.append(cur_loss)
        val_roc.append(roc_curr)
        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch: {epoch + 1:04d}; train_loss= {cur_loss:.5f}; val_roc= {roc_curr:.5f}; val_ap= {ap_curr:.5f}; time= {time.time() - t:.5f}")
        if roc_curr > best_val_auc:
            torch.save(model.state_dict(), best_model_path)
            best_val_auc = roc_curr
        # Early stopping
        if roc_curr < the_last_roc:
            trigger_times += 1
            if trigger_times % 10 == 0:
                print('trigger times:', trigger_times)
            if trigger_times >= patience:
                print('Early stopping!\nStart the test process.')
                break
        else:
            trigger_times = 0
        the_last_roc = roc_curr

    print("Optimization Finished!")
    ### Ploting Learning #####
    if plot_training:
        plot_learning(train_loss, val_roc, plot_learning_file)
    ##########################
    return adj_norm


def evaluate_model(model, adj_norm, features,  edges, edges_false,best_model_path=None):
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
    recovered, mu, logvar = model(features, adj_norm)
    # hidden_emb = mu.data.numpy()
    auc, ap = get_roc_score(mu.data.numpy(), edges, edges_false)
    print(f'Test ROC score: {auc:.5} ; Test AP score: {ap:.5}')
    return auc, ap
