import numpy as np
import networkx as nx
from src.utils.project_constants import *

SIGNIFICANCE_BASED_LABELING = True

def split2groups(diff, required_diff):

    significant_diffs = dict([item for item in diff['pairwise_comparisons'].items() if item[1]['significant_diff']])
    logs2ps = {}
    for item_id in significant_diffs:
        if item_id[0] not in logs2ps:
            logs2ps[item_id[0]] = significant_diffs[item_id]['p1']
        if item_id[1] not in logs2ps:
            logs2ps[item_id[1]] = significant_diffs[item_id]['p2']
    logs_sorted_by_proportion = sorted(logs2ps.items(), key=lambda x: x[1])

    groups = significance_based_distance(logs_sorted_by_proportion, significant_diffs) if SIGNIFICANCE_BASED_LABELING \
        else distance_based_grouping(logs_sorted_by_proportion, required_diff)

    labels = []
    means = []
    for gr in groups:
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind': float_formatter})
        means.append(np.round(np.mean([v[1] for v in gr]), 2))
        labels.append([v[0] for v in gr])

    return labels, means


def significance_based_distance(logs_sorted_by_proportion, significant_diffs):

    groups, grp = [], []
    for log_tup in logs_sorted_by_proportion:
        for log_tup2 in grp:
            significant_diff = (log_tup[0], log_tup2[0]) in significant_diffs \
                               or (log_tup2[0], log_tup[0]) in significant_diffs
            if significant_diff:
                groups.append(grp)
                grp = []
                break
        grp.append(log_tup)
    if len(grp):
        groups.append(grp)
    return groups

def distance_based_grouping(logs_sorted_by_proportion, required_diff):

    current_group = [logs_sorted_by_proportion[0]]
    groups = [current_group]
    i = 1
    while i < len(logs_sorted_by_proportion):
        if logs_sorted_by_proportion[i][1] > logs_sorted_by_proportion[i - 1][1] + required_diff:
            current_group = [logs_sorted_by_proportion[i]]
            groups.append(current_group)
        else:
            current_group.append(logs_sorted_by_proportion[i])
        i += 1
    return groups

def overlay_differences_over_graph(g, sig_diffs, delta, add_test_info=True, coloring_scheme=False):

    if len(sig_diffs) == 0:
        return
    colors, stats, statstics_val = {}, {}, {}
    import networkx as nx
    labels = nx.get_node_attributes(g, 'label')
    edges = nx.edges(g)
    d = dict([(labels[l], l) for l in labels])
    for diff in sig_diffs:
        source = tuple([x for x in diff[SOURCE_ATTR_NAME]])
        target = tuple([x for x in diff[TARGET_ATTR_NAME]])
        if source not in d or target not in d:
            continue
        src_id = d[source]
        trg_id = d[target]
        if (src_id, trg_id) not in edges:
            continue
        colors[(src_id, trg_id)] = 'red'
        if 'different_ids' in diff: ## handle n2KDiff results
            groups, means = split2groups(diff, delta)
            means_str = ", ".join([str(round(mean, 2)) for mean in means])
            grps_str = [str(sorted(gr)) for gr in groups]
            stats[(src_id, trg_id)] = str(
                grps_str) + ": " + means_str
            # stats[(src_id, trg_id)] = str(round(float(diff[PVALUE_ATTR_NAME]), 2)) + ": " +  str(grps_str) + " " + means_str
            # import math
            # most_sig_diff = sorted(diff['pairwise_comparisons'].items(), key=lambda test: 1 if math.isnan(test[1]['pvalue']) else test[1]['pvalue'])[0]
            # stats[(src_id, trg_id)] = str(round(float(diff[PVALUE_ATTR_NAME]), 2)) + ": " + str(most_sig_diff[0]) + \
            #                           " - " + str(round(most_sig_diff[1]['p1'], 2)) + ", " + str(round(most_sig_diff[1]['p2'], 2))

            # stats[(src_id, trg_id)] = str(diff['different_ids']) + \
            #                       "; pvalue:" + str(round(float(diff[PVALUE_ATTR_NAME]), 2)) + \
            #                       "; transitions per log:" + str(diff[EXPERIMENTS_PER_LOG_ATTR_NAME])
        else: ## handle s2KDiff results
            stats[(src_id, trg_id)] = "p1:" + str(round(diff[P1_ATTR_NAME], 2)) + "; p2:" + str(round(diff[P2_ATTR_NAME], 2)) + \
                                  "; m1:" + str(diff[M1_ATTR_NAME]) + "; m2:" + str(diff[M2_ATTR_NAME]) + \
                                  "; pvalue:" + str(diff[PVALUE_ATTR_NAME])
        statstics_val[(src_id, trg_id)] = np.log(float(diff[STATISTICS_ATTR_NAME])) if float(diff[STATISTICS_ATTR_NAME]) > 1 else float(diff[STATISTICS_ATTR_NAME])
        if coloring_scheme:
            some_diff = list(diff['pairwise_comparisons'].keys())[0]
            p1 = diff['pairwise_comparisons'][some_diff][P1_ATTR_NAME]
            p2 = diff['pairwise_comparisons'][some_diff][P2_ATTR_NAME]
            colors[(src_id, trg_id)] = 'red' if p1 > p2 else 'green'
    nx.set_edge_attributes(g, colors, 'color')
    edge_labels = nx.get_edge_attributes(g, 'label')
    min_statistic = min(statstics_val.items(), key=lambda v: v[1])[1]
    max_statistic = max(statstics_val.items(), key=lambda v: v[1])[1]
    pen_widths = {}
    edges2remove = []
    for e in edge_labels:
        if e in stats:
            if add_test_info:
                # edge_labels[e] = str(edge_labels[e]) + "_" + str(stats[e])
                edge_labels[e] = str(stats[e])
            pen_widths[e] = \
                1 + min(1.5 * (1-((statstics_val[e] - min_statistic) / (max_statistic - min_statistic))), 3)
        else:
            edges2remove.append(e)
    nx.set_edge_attributes(g, edge_labels, 'label')
    nx.set_edge_attributes(g, pen_widths, 'penwidth')

    filter_nodes = False
    if filter_nodes:
        print('------ graph filtering ------')
        node_labels = nx.get_node_attributes(g, 'label')
        for node in node_labels:
            if not len(node_labels[node]):
                continue
            node_label = node_labels[node][0]
            print(node_labels[node])
            # if node_label in ['homepage', 'search', 'sales_anncs'] or 'sales_page' in node_label:
            #     continue
            if node_label in ['sales_anncs', 'sales_page, page_1', 'sales_page, page_2', 'sales_page, page_3']:
                continue
            g.remove_node(node)
    # for e in edges2remove:
    #     if e[0] in g.nodes() and e[1] in g.nodes():
    #         g.remove_edge(e[0], e[1])
    # edge_labels = nx.get_edge_attributes(g, 'label')
    # for e in edge_labels:
    #     if "edge_labels[e]
    print('------ ------ ------ ------')
    return g


def write2file(g, path):

    def remove_attribute(G, tnode, attr):
        G.node[tnode].pop(attr, None)

    g = g.copy()
    for n in g.nodes():
        remove_attribute(g, n, "contraction")
    for e in g.edges:
        del g.get_edge_data(e[0], e[1])[TRACES_ATTR_NAME]
    nx.drawing.nx_pydot.write_dot(g, path)


