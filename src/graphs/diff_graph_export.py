import numpy as np

def overlay_differences_over_single_graph(g, sig_diffs):

    colors, stats, statstics_val = {}, {}, {}
    import networkx as nx
    labels = nx.get_node_attributes(g, 'label')
    d = dict([(labels[l], l) for l in labels])
    for diff in sig_diffs:
        source = tuple([x.lower() for x in diff['source']])
        target = tuple([x.lower() for x in diff['target']])
        src_id = d[source]
        trg_id = d[target]
        colors[(src_id, trg_id)] = 'red'
        if 'different_ids' in diff: ## handle n2KDiff results
            stats[(src_id, trg_id)] = str(diff['different_ids']) + \
                                  "; pvalue:" + str(round(float(diff['pvalue']), 2)) + \
                                  "; transitions per log:" + str(diff['experiments_per_log'])
        else: ## handle s2KDiff results
            stats[(src_id, trg_id)] = "p1:" + str(round(diff['p1'], 2)) + "; p2:" + str(round(diff['p2'], 2)) + \
                                  "; m1:" + str(diff['m1']) + "; m2:" + str(diff['m2']) + \
                                  "; pvalue:" + str(diff['pvalue'])
        statstics_val[(src_id, trg_id)] = np.log(float(diff['statistic'])) if float(diff['statistic']) > 1 else float(diff['statistic'])
    nx.set_edge_attributes(g, colors, 'color')
    edge_labels = nx.get_edge_attributes(g, 'label')
    min_statistic = min(statstics_val.items(), key=lambda v: v[1])[1]
    max_statistic = max(statstics_val.items(), key=lambda v: v[1])[1]
    pen_widths = {}
    for e in edge_labels:
        if e in stats:
            edge_labels[e] = str(edge_labels[e]) + "_" + str(stats[e])
            pen_widths[e] = \
                1 + min(3 * ((statstics_val[e] - min_statistic) / (max_statistic - min_statistic)), 5)
    nx.set_edge_attributes(g, edge_labels, 'label')
    nx.set_edge_attributes(g, pen_widths, 'penwidth')
    return g