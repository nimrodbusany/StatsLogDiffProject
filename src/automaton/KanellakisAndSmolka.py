# This is Ariel Hoffman's implementation of Kanellakis and Smolka's
# coarse partitioning system.

# To determine whether two machines are bisimilar, we can just find
# both their coarsest partition and then relabel the nodes accordingly.
# If they are not bisimilar, this will not work.

## credits: ArielCHoffman
## link to source: https://github.com/ArielCHoffman/BisimulationAlgorithms


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from src.graphs.example_graphs import *


class KanellakisAndSmolka:
    def __init__(self, labels):
        self.labels = labels

    def getNeighborsGivenLabels(self, node, label, Q):

        neighbors = list(Q.neighbors(node))
        labels = []
        for n in neighbors:
            if (node, n) not in Q.edges():
                continue
            for edge_dict in Q.get_edge_data(node, n).values():
                labels.append(edge_dict["label"])
        trueNeighbors = []
        for neighbor, a in zip(neighbors, labels):
            if a == label:
                trueNeighbors.append(neighbor)
        return trueNeighbors

    # Here, Q is the graph,
    # a is the action
    # Notice that this is many more lines than using the simple splitting
    # Paige and Tarjan take advantage of.
    def split(self, blocks, a, Q, blockIdFromNode):
        # This is where we'll store the new set of blocks.
        newBlocks = []

        # Now we go ahead and split every block.
        B1 = []
        B2 = []
        for block in blocks:
            initialNode = block[0]  # This should be a node id.
            # Find the initaial neighbors
            neighbors = self.getNeighborsGivenLabels(initialNode, a, Q)
            neighborBlocks = [blockIdFromNode[n] for n in neighbors]
            initialSetOfDestBlocks = set(neighborBlocks)

            # Now we initialize our two new blocks:
            blockB1 = []
            blockB2 = []
            for node in block:
                neighbors = self.getNeighborsGivenLabels(node, a, Q)
                neighborBlocks = [blockIdFromNode[n] for n in neighbors]
                setOfDestBlocks = set(neighborBlocks)
                if (setOfDestBlocks == initialSetOfDestBlocks):
                    blockB1.append(node)
                else:
                    blockB2.append(node)
            B1.append(blockB1)
            if blockB2 != []:
                B2.append(blockB2)
                # We now have B1 and B2 for every block.

        # Before we exit, we need to reinitialize our look up dictionary for the block numbers:
        return B1 + B2

    # This will take the graph of interest and output the desired partition.
    # You also need to supply the labels of interest to you.
    def compute_coarsest_partition(self, Q, blocks=None, plot=False):
        blockIdFromNode = dict.fromkeys(Q.nodes(), 0)  # Each node has a blockID.
        prevLen = 0
        pos = nx.spring_layout(Q)
        if not blocks:
            blocks = [list(Q.nodes())]
        while prevLen != len(blocks):
            # print('blocks:', blocks)
            prevLen = len(blocks)
            for label in self.labels:
                if (plot):
                    self.plotGraph(Q, blocks, pos)
                blocks = self.split(blocks, label, Q, blockIdFromNode)
                # print('split by label: ', label, ":", blocks)
                for i in range(len(blocks)):
                    for node in blocks[i]:
                        blockIdFromNode[node] = i
        return blocks

    # This plots the graph. The arrows are color coded according to label
    # and the nodes are color coded according to partition. The partition value
    # will depend on the global variable. Initially, there will be only one
    # partition and all the nodes will have the same color.
    def plotGraph(self, Q, blocks=None, pos=None):

        if blocks == None:
            blocks = [[n] for n in Q.nodes()]
        numOfLabels = len(self.labels)
        numOfBlocks = len(blocks)

        plt.figure(1)
        if not pos:
            pos = nx.spring_layout(Q)
        nodeColors = cm.rainbow(np.linspace(0, 1, numOfBlocks))
        edgeColors = cm.rainbow(np.linspace(0, 1, numOfLabels))
        for i in range(len(blocks)):
            nx.draw_networkx_nodes(Q, pos, nodelist=blocks[i], node_color=nodeColors[i])
        for i in range(numOfLabels):
            acts = []
            for edge in Q.edges():
                for edge_dict in Q.get_edge_data(*edge).values():
                    if (edge_dict["label"] == self.labels[i]):
                        acts.append(edge)
            nx.draw_networkx_edges(Q, pos, edgelist=acts, edge_color=[edgeColors[i]] * len(acts))
        plt.show()

    # This verifies that the two graphs are bisimilar.
    def isBisimilar(self, Q1, Q2):
        Q = nx.union_all([Q1, Q2], rename=('G-', 'H-'))
        blocks = self.compute_coarsest_partition(Q)
        # Confirms that there is at least one of each type of node in each partition.
        # If there is, we have bisimilarity. Otherwise, we don't.
        for block in blocks:
            if not any('H' in nodeName for nodeName in block):
                return False
            if not any('G' in nodeName for nodeName in block):
                return False
        return True


# Every graph must have edges and edge labels.
# In this case, edge labels will be stored as

def example1():
    # EXAMPLE 1
    # Basic tests:
    # The following test is example 1, shown in class
    Q = nx.MultiGraph()
    # Q = nx.DiGraph()
    Q.add_edge(1, 2, label=1)
    Q.add_edge(1, 3, label=2)

    Q.add_edge(1, 4, label=3)
    Q.add_edge(3, 4, label=1)
    Q.add_edge(3, 5, label=2)
    labels = [1, 2, 3]

    k = KanellakisAndSmolka(labels)
    k.compute_coarsest_partition(Q, plot=True)

def example2():
    # EXAMPLE 2
    # the following is the bisimulation example:
    S = nx.MultiGraph()
    S.add_edge(0, 1, label=1)
    S.add_edge(1, 2, label=2)
    S.add_edge(0, 2, label=1)
    S.add_edge(2, 2, label=2)
    labels = [1, 2]

    T = nx.MultiGraph()
    T.add_edge(0, 1, label=1)
    T.add_edge(1, 1, label=2)

    k = KanellakisAndSmolka(labels)
    # Verifies Bisimilarity.
    print(k.isBisimilar(S, T))
    # Demos the coarsest partition functionality again.
    Q = nx.disjoint_union_all([S, T])
    k.compute_coarsest_partition(Q, plot=True)

def example4():
    # EXAMPLE 2
    # the following is the bisimulation example:
    S = nx.MultiDiGraph()
    S.add_node(1)
    S.add_node(2)
    S.add_node(3)
    S.add_node(4)
    S.add_node(5)
    S.add_node(6)
    S.add_edge(1, 2, action=1, key=1)
    S.add_edge(1, 3, action=1, key=2)
    S.add_edge(2, 4, action=1, key=3)
    S.add_edge(2, 6, action=1, key=4)
    S.add_edge(3, 4, action=1, key=5)
    S.add_edge(3, 5, action=1, key=6)

    labels = [1]
    from PaigeAndTarjan import PaigeAndTarjan
    k = PaigeAndTarjan(labels)

    plt.figure(1)
    pos = nx.spring_layout(S)
    nx.draw_networkx_nodes(S, pos, nodelist=S.nodes)
    nx.draw_networkx_edges(S, pos)
    plt.show()

    print(k.getCoarsestPartition(S, plot=True))



def example3():

    S, labels = get_k_struct()
    k = KanellakisAndSmolka(labels)
    blocks = k.compute_coarsest_partition(S.reverse())
    print('reverse:', blocks)
    blocks = k.compute_coarsest_partition(S)
    print('forward:', blocks)
    pos = nx.spring_layout(S)
    k.plotGraph(S, blocks, pos)


def coarsen_graph(S, blocks):

    D = nx.MultiDiGraph()
    node2block = {}
    for i in range(len(blocks)):
        block = blocks[i]
        D.add_node(i)
        for n in block:
            node2block[n] = i

    edge_attribute_dict = {}
    processed_edges = set()
    edge_id = 0
    for e in S.edges():
        edge_id+=1
        for edge_dict in S.get_edge_data(e[0], e[1]).values():
            b1 = node2block[e[0]]
            b2 = node2block[e[1]]
            a = edge_dict["label"]
            if (b1, a, b2) not in processed_edges:
                D.add_edge(b1, b2, edge_id)
                edge_attribute_dict[(b1, b2, edge_id)] = a
                processed_edges.add((b1, a, b2))
    nx.set_edge_attributes(D, edge_attribute_dict, 'label')
    return D

if __name__ == '__main__':

    example3()
    exit()
    # example4()
    S, labels = get_k_struct()
    k = KanellakisAndSmolka(labels)
    S_reverse = S.reverse()
    blocks = k.compute_coarsest_partition(S_reverse)
    S_coarsen = coarsen_graph(S, blocks)
    # k.plotGraph(S_coarsen)
    nx.drawing.nx_pydot.write_dot(S_coarsen, 'out.dot')
    blocks = k.compute_coarsest_partition(S_coarsen)
    S_coarsen = coarsen_graph(S_coarsen, blocks)
    nx.drawing.nx_pydot.write_dot(S_coarsen, 'out2.dot')
    # k.plotGraph(S_coarsen)
    print(len(S.nodes), len(S_coarsen.nodes), len(blocks))
    print("blocks reversed graph first:", blocks)

    # S, labels = get_k_struct()
    # k = KanellakisAndSmolka(labels)
    # blocks = k.getCoarsestPartition(S)
    # S_coarsen = coarsen_graph(S, blocks)
    # S_coarsen = S_coarsen.reverse()
    # blocks = k.getCoarsestPartition(S_coarsen)
    # S_coarsen = coarsen_graph(S_coarsen, blocks)
    # k.plotGraph(S_coarsen)
    #
    # print(len(S.nodes), len(S_coarsen.nodes), len(blocks))
    # print("blocks reversed graph first:", blocks)
    # print("blocks graph first:", blocks)
    #

    # example3_reverse()
    # example3()
