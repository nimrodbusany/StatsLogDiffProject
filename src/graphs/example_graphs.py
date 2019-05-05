import networkx as nx

def get_k_struct():

    S = nx.MultiDiGraph()
    S.reverse()
    S.add_edge(0, 1, label='f')

    S.add_edge(1, 2, label='a')
    S.add_edge(1, 3, label='a')
    S.add_edge(1, 4, label='a')

    S.add_edge(2, 5, label='b')
    S.add_edge(3, 6, label='b')
    S.add_edge(4, 7, label='b')

    S.add_edge(5, 8, label='c')
    S.add_edge(6, 9, label='d')
    S.add_edge(7, 10, label='e')

    S.add_edge(8, 11, label='g')
    S.add_edge(9, 11, label='g')
    S.add_edge(10, 11, label='g')

    labels = ['a', 'b', 'c', 'd', 'e', 'f','g']

    return S, labels
