import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import networkx as nx


def plot_graph(G):
    pos = nx.spring_layout(G)  # G is my graph
    nx.draw(G, pos, node_color='dodgerblue',edge_color='black', width=1.5, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("graph.png", dpi=500, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)


def plot_learning(train_loss, val_roc, file_name):
    fig, ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train loss', color=color)
    ax1.plot(np.arange(len(train_loss)), train_loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color='blue'
    ax2.set_ylabel('Validation ROC', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(len(val_roc)), val_roc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(join('Output', file_name+'.png'), dpi=300)
    plt.savefig(join('Output', file_name+'.eps'), format='eps')


