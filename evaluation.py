import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_roc_score(emb, edges_pos, edges_neg):
    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))

    edges_neg= edges_neg.astype(int)
    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

## Old original function- delete later
def get_roc_score_old(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    edges_neg= edges_neg.astype(int)
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def pr_curve(y_label, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_label = y_label[desc_score_indices]
    tp, fp, tps, fps = 0., 0., [], []
    for i in y_label:
        if i == 1:
            tp += 1
        else:
            fp += 1
        fps.append(fp)
        tps.append(tp)


    precision, recall = [], []
    for f, t in zip(fps, tps):
        precision.append(t / (t + f))
        recall.append(t / tps[-1])
    return precision, recall


def evalution(adj_rec, train_pos, test_pos):
    num = adj_rec.shape[0]
    x, y = np.triu_indices(num, k=1)

    c_set = set(zip(x, y)) - \
            set(zip(train_pos[:, 0], train_pos[:, 1])) - set(zip(train_pos[:, 1], train_pos[:, 0]))

    inx = np.array(list(c_set))
    Y = np.zeros((num, num))
    Y[test_pos[:, 0], test_pos[:, 1]] = 1
    Y[test_pos[:, 1], test_pos[:, 0]] = 1
    labels = Y[inx[:, 0], inx[:, 1]]
    val = adj_rec[inx[:, 0], inx[:, 1]]

    fpr, tpr, throc = roc_curve(labels, val)
    auc_val = auc(fpr, tpr)
    # prec, rec, thpr = precision_recall_curve(labels, val)
    prec, rec = pr_curve(labels, val)
    aupr_val = auc(rec, prec)

    f1_val = 0
    for i in range(len(prec)):
        if (prec[i] + rec[i]) == 0:
            continue
        f = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
        if f > f1_val:
            f1_val = f

    return auc_val, aupr_val, f1_val