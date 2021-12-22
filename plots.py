import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import networkx as nx


VGAE = [0.87138, 0.8112, 0.814466, 0.8929]
RF = [0.815744, 0.73854, 0.78527, 0.91263]

index = np.arange(len(VGAE))
bar_width = 0.35

fig, ax = plt.subplots()
summer = ax.bar(index, VGAE, bar_width,
                label="VGAE", color='#B266FF')

winter = ax.bar(index+bar_width, RF,
                 bar_width, label="RF", color='#66B2FF')

ax.set_xlabel('GI dataset')
ax.set_ylabel('ROC')
ax.set_title('GI Results')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["Costanzo-09", "Costanzo-16", "K562", "SynLethDB"])
ax.legend()

plt.show()
def plot2():
    fig = plt.figure()
    # fig, ax = plt.subplots(figsize=(10, 5))
    title = ['GNAE', 'VGAE GI', 'VGAE PPI', 'Random features', 'Random connections', 'RF']
    values = [0.846, 0.767, 0.731, 0.669, 0.672, 0.764]
    x_pos = np.arange(len(values))
    my_colors = ['indigo','m','g','b','k','c']
    plt.bar(x_pos, values, color=my_colors)
    # ax.bar(x_pos, values)
    for index, data in enumerate(values):
        plt.text(x=index, y=data, s=f"{data}", fontdict=dict(fontsize=10))
    # Create names on the x-axis
    plt.xticks(x_pos, title)
    plt.title("Drug Sensitivity")
    plt.ylabel('AUC')
    plt.xlabel('Dataset')
    plt.show()

def plot1():
    fig, ax = plt.subplots()
    ind = list(np.arange(1, 3))
    width = 0.4
    # show the figure, but do not block
    plt.show(block=False)

    go =[0.4154, 0.2944]
    random =[0.0099, 0.01825]

    plt.bar([i - (width/2) for i in ind], go, width,alpha=0.5, color='b')
    plt.bar([i + (width/2) for i in ind], random, width, color='g')


    ax.set_xticks(ind)
    ax.set_xticklabels(['K562', 'Jurkat'])
    ax.set_ylim([0.0, 0.5])
    ax.set_ylabel('Pearson Correlation (predicted, actual)')
    ax.set_title('Human GI - Regression')

    plt.legend(['$F_{GO}$', '$F_{Random  GO}$'], loc='upper right')

    plt.savefig("Human GI - Regression")
    plt.show()

    fig, ax = plt.subplots()
    ind = list(np.arange(1, 3))
    width = 0.4
    # show the figure, but do not block
    plt.show(block=False)

def plot_graph(G):
    pos = nx.spring_layout(G)  # G is my graph
    nx.draw(G, pos, node_color='dodgerblue',edge_color='black', width=1.5, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.savefig("graph.png", dpi=500, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)
