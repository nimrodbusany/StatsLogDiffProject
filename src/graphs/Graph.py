import networkx as nx
from itertools import chain
import numpy as np
from collections import OrderedDict


def map_weights_to_widths(edges2weights):

    weights = np.array(list(edges2weights.values()))
    if not np.any(weights):
        return
    values = []
    for i in range(10, 101, 10):
        values.append(np.percentile(weights, i))
    weights2widths = {}
    for e, w in edges2weights.items():
        i = 0
        while w > values[i]:
            i += 1
        weights2widths[e[0], e[1]] = i * 0.5 + 0.5
    return weights2widths

class OrderedDiGraph(nx.DiGraph):
    """
    nx.DiGraph that retains ordering when iterating on it
    """
    adjlist_outer_dict_factory = OrderedDict
    adjlist_inner_dict_factory = OrderedDict
    node_dict_factory = OrderedDict

class OrderedGraph(nx.DiGraph):
    """
    nx.Graph that retains ordering when iterating on it
    """
    adjlist_outer_dict_factory = OrderedDict
    adjlist_inner_dict_factory = OrderedDict
    node_dict_factory = OrderedDict


class DGraph:
    def __init__(self, nx_graph=None):
        self.dgraph = OrderedDiGraph() if nx_graph is None else nx_graph

    def add_node(self, node, label=None, **attr):
        self.dgraph.add_node(node,label=label, attr = attr)

    def add_edge(self, node1, node2, weight=None):
        if(weight == None):
            self.dgraph.add_edge(node1, node2)
        else:
            self.dgraph.add_edge(node1, node2, weight=weight)

    def add_edge_attribute(self, node1, node2, attribute_name, attribute_value):
        self.dgraph.edges[node1, node2][attribute_name] = attribute_value

    def nodes(self):
        return self.dgraph.nodes()

    def get_edge_data(self, edge):
        return self.dgraph.get_edge_data(edge[0], edge[1])

    def edges(self, data=True):
        return self.dgraph.edges(data=data)
    
    def edges_of_node(self, node): 
        return self.dgraph.edges(node)

    def in_edges(self, node):
        return self.dgraph.in_edges(node)

    def out_edges(self, node):
        return self.dgraph.out_edges(node)


    def draw(self):
        import matplotlib.pyplot as plt
        nx.draw(self.dgraph)
        plt.show()

    def number_of_nodes(self):
        return self.dgraph.number_of_nodes()
    
    def number_of_edges(self):
        return self.dgraph.size()


    def node_attr(self, node, attr):
        return self.dgraph.nodes[node].get(attr,'')

    def write_dot(self, path, produce_png= False):
        print("write_dot called")
        nx.drawing.nx_pydot.write_dot(self.dgraph, path)
        if produce_png:
            print("from dot to png")
            import subprocess, os
            fname = os.path.basename(path)
            if fname.endswith('.dot'):
                fname = fname[:-4]
            folder = os.path.dirname(path)
            outpath = folder + "/" + fname
            subprocess.run('dot -Tpng -o ' + outpath + '.png ' + path)


    def adjacency_matrix(self, node_list=None):
        # order nodes' row,column according to list or according to dgraph.nodes() order
        if node_list is None:
            node_list = self.nodes()

        adj_mat = np.zeros(shape=(len(self.nodes()),len(self.nodes())))
        i = 0
        node_dict = {node : i for i, node in enumerate(node_list)}
            
        for edge in self.edges():
            adj_mat[node_dict[edge[0]]][node_dict[edge[1]]] += int(edge[2].get('weight',1))

        return adj_mat

    def subgraph(self, vertices):
        return self.dgraph.subgraph(vertices)    


    def max_in_out_degree(self):
        ":returns: maximal in- or out-degree in the graph"
        maxdeg = 0
        for node in self.dgraph.nodes():
            maxdeg = max(maxdeg,max(self.dgraph.in_degree(node), self.dgraph.out_degree(node)))
        return maxdeg

    def number_of_components(self):
        ":returns: number of strongly-connected components in the graph"
        return sum(1 for _ in nx.strongly_connected_components(self.dgraph))


    @staticmethod
    def read_dot(path):
        multigraph = nx.drawing.nx_pydot.read_dot(path)
        nx_graph = OrderedDiGraph()
        # create the graph ordered, for consistency when running algorithms
        for node, data in sorted(multigraph.nodes(data=True)):
            if "label" not in data:
                data["label"] = ""
            nx_graph.add_node(node, **data)

        ## edges may be spread across the dot model
        edges_dic = {}
        for src, dst, data in multigraph.edges(data=True):
            attributes = edges_dic.get((src, dst), [])
            if "label" not in data:
                data["label"] = ""
            attributes.append(data)
            edges_dic[(src, dst)] = attributes

        ## prepare normalized weights; TODO: some arbitrariness may occur when only some of the edges are weighted
        weights = {}
        for edge, data in edges_dic.items():
            hasWeight = any([d.get("weight") for d in data])
            if hasWeight:
                weights[(edge[0], edge[1])] = sum([float(d.get("weight", 0.0)) for d in data])
        weights2widths = map_weights_to_widths(weights)

        ## unify edges data
        for edge, data in edges_dic.items():
            has_weight_attr = False
            has_penwidth_attr = False
            agg_dic = {}
            for attr in data[0]:
                agg_value = ""
                if attr == "label":
                    agg_value = ";\n".join([d.get(attr, "").strip("\"").strip("\'") for d in data])
                elif attr == "weight":
                    has_weight_attr = True
                    agg_value = weights[(edge[0], edge[1])]
                elif attr == "penwidth":
                    has_penwidth_attr = True
                else:
                    agg_value = data[0].get(attr, '')
                agg_dic[attr] = agg_value

            if not has_weight_attr: ## if no edge weight is defined, assign eqaul weights to all edges
                agg_dic['weight'] = max(1, len(data)) ## edge weight = number of transition in multigraph
                if not has_penwidth_attr:
                    agg_dic['penwidth'] = min(agg_dic['weight'], 5.5)
                    has_penwidth_attr = True

            if has_penwidth_attr is False:
                agg_dic['penwidth'] = weights2widths[(edge[0], edge[1])]

            nx_graph.add_edge(edge[0], edge[1], **agg_dic)

        # deal with ""-encapsulated strings in properties
        # e.g. java.net.DatagramSocket.dot
        for item, attrs in chain(nx_graph.nodes.items(), nx_graph.edges.items()):
            for key in ('label', 'style'):
                if key in attrs:
                    # remove surrounding double-quotes
                    attrs[key] = attrs[key].strip().strip('"').strip()

        return DGraph(nx_graph)

    def project(self, vertices):  
        return DGraph(self.subgraph(vertices))     

    def get_sink_nodes(self):
        sink_nodes = []
        for node_item in self.dgraph.nodes(data=True):
            if len(self.dgraph.out_edges(node_item[0])) == 0:
                sink_nodes.append(node_item)
        return sink_nodes

    def get_initial_nodes(self):
        inital_nodes = []
        for node_item in self.dgraph.nodes(data=True):
            if len(self.dgraph.in_edges(node_item[0])) == 0:
                inital_nodes.append(node_item)
        return inital_nodes


