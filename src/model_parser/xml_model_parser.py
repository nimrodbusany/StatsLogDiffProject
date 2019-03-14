import xml.etree.ElementTree as ET
import networkx as nx

from src.graphs.graphs import DGraph
from src.utils.project_constants import *

def from_zeller_xml_to_dgraph(path):

    def set_shape(node_name):
        if node_name.strip() == 'start':
            return "box"

        if node_name.strip() == 'TERMINAL':
            return "doublecircle"
        return "circle"

    tree = ET.parse(path)
    root = tree.getroot()
    nodes = root.findall('node')
    edges = root.findall('edge')
    nodes_arr = []
    transitions_arr = []
    shapes = {}
    for node in nodes:
        node_name = node.get('id')
        nodes_arr.append(node_name)
        shapes[node_name] = set_shape(node_name)

    for edge in edges:
        source = edge.get(SOURCE_ATTR_NAME)
        target = edge.get(TARGET_ATTR_NAME)
        label = edge.find('data').text
        transitions_arr.append((source, target, label))

    graph = DGraph()
    for n in nodes_arr:
        graph.add_node(n, label=n)

    probs = {}
    trans_labels = {}
    for t in transitions_arr:
        graph.add_edge(t[0], t[1])
        if (t[0], t[1]) in trans_labels:
            raise AssertionError('does not support multiple edges between two nodes')
        trans_labels[(t[0], t[1])] = t[2]

    nx.set_node_attributes(graph.dgraph, shapes, 'shape')
    nx.set_edge_attributes(graph.dgraph, trans_labels, 'label')
    nx.set_edge_attributes(graph.dgraph, probs, 'prob')
    return graph


def from_david_xml_to_dgraph(path):
    def set_shape(node_name, start_node):
        if node_name == start_node.text:
            return "box"
        if node_name.strip() == 'TERMINAL':
            return "doublecircle"
        return "circle"

    tree = ET.parse(path)
    root = tree.getroot()
    nodes = root.find('nodes')
    initial_node = root.find('startnode')
    print('Init', initial_node.text)
    nodes_arr = []
    transitions_arr = []
    shapes = {}
    for node in nodes:
        node_name = node.get('name')
        nodes_arr.append(node_name)
        shapes[node_name] = set_shape(node_name, initial_node)
        gotos_obj = node.find('gotos')
        if not gotos_obj:
            continue
        gotos = gotos_obj.findall('goto')
        for goto in gotos:
            prob = goto.get('prob')
            label = goto.get('out')
            trg_ = goto.text
            transitions_arr.append((node_name, trg_, label, prob))

    graph = DGraph()
    for n in nodes_arr:
        graph.add_node(n, label=n)
    probs = {}
    trans_labels = {}
    for t in transitions_arr:
        graph.add_edge(t[0], t[1])
        if (t[0], t[1]) in trans_labels:
            raise AssertionError('does not support multiple edges between two nodes')
        trans_labels[(t[0], t[1])] = t[2]
        probs[(t[0], t[1])] = t[3]
    nx.set_node_attributes(graph.dgraph, shapes, 'shape')
    nx.set_edge_attributes(graph.dgraph, trans_labels, 'label')
    nx.set_edge_attributes(graph.dgraph, probs, 'prob')
    return graph

if __name__ == '__main__':
    path = 'C:/Users/USER/PycharmProjects/StatsLogDiffProject/ktails_models/david/jfreechart.xml'
    output_path = 'jfreechart.dot'
    graph = from_david_xml_to_dgraph(path)
    # graph = from_zeller_xml_to_dgraph(path)
    graph.write_dot(output_path)
