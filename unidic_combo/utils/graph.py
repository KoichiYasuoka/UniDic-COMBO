"""Based on https://github.com/emorynlp/iwpt-shared-task-2020."""
from typing import List

import numpy as np


def sdp_to_dag_deps(arc_scores, rel_scores, tree_tokens: List, root_idx=0, vocab_index=None) -> None:
    # adding ROOT
    tree_heads = [0] + [t["head"] for t in tree_tokens]
    graph = adjust_root_score_then_add_secondary_arcs(arc_scores, rel_scores, tree_heads,
                                                      root_idx)
    for i, (t, g) in enumerate(zip(tree_heads, graph)):
        if not i:
            continue
        rels = [vocab_index.get(x[1], "root") if vocab_index else x[1] for x in g]
        heads = [x[0] for x in g]
        head = tree_tokens[i - 1]["head"]
        index = heads.index(head)
        deprel = tree_tokens[i - 1]["deprel"]
        deprel = deprel.split('>')[-1]
        # TODO - Consider if there should be a condition,
        # It doesn't seem to make any sense as DEPS should contain DEPREL
        # (although sometimes with different/more detailed label)
        # if len(heads) >= 2:
        #     heads.pop(index)
        #     rels.pop(index)
        deps = '|'.join(f'{h}:{r}' for h, r in zip(heads, rels))
        tree_tokens[i - 1]["deps"] = deps
        tree_tokens[i - 1]["deprel"] = deprel
    return


def adjust_root_score_then_add_secondary_arcs(arc_scores, rel_scores, tree_heads, root_idx):
    if len(arc_scores) != tree_heads:
        arc_scores = arc_scores[:len(tree_heads)][:len(tree_heads)]
        rel_scores = rel_scores[:len(tree_heads)][:len(tree_heads)]
    # Self-loops aren't allowed, mask with 0. This is an in-place operation.
    np.fill_diagonal(arc_scores, 0)
    parse_preds = np.array(arc_scores) > 0
    parse_preds[:, 0] = False  # set heads to False
    rel_scores[:, :, root_idx] = -float('inf')
    return add_secondary_arcs(arc_scores, rel_scores, tree_heads, root_idx, parse_preds)


def add_secondary_arcs(arc_scores, rel_scores, tree_heads, root_idx, parse_preds):
    if not isinstance(tree_heads, np.ndarray):
        tree_heads = np.array(tree_heads)
    dh = np.argwhere(parse_preds)
    sdh = sorted([(arc_scores[x[0]][x[1]], list(x)) for x in dh], reverse=True)
    graph = [[] for _ in range(len(tree_heads))]
    rel_pred = np.argmax(rel_scores, axis=-1)
    for d, h in enumerate(tree_heads):
        if d:
            graph[h].append(d)
    for s, (d, h) in sdh:
        if not d or not h or d in graph[h]:
            continue
        try:
            path = next(_dfs(graph, d, h))
        except StopIteration:
            # no path from d to h
            graph[h].append(d)
    parse_graph = [[] for _ in range(len(tree_heads))]
    num_root = 0
    for h in range(len(tree_heads)):
        for d in graph[h]:
            rel = rel_pred[d][h]
            if h == 0:
                rel = root_idx
                assert num_root == 0
                num_root += 1
            parse_graph[d].append((h, rel))
        parse_graph[d] = sorted(parse_graph[d])
    return parse_graph


def _dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path + [next_state]))


def restore_collapse_edges(tree_tokens):
    # https://gist.github.com/hankcs/776e7d95c19e5ff5da8469fe4e9ab050
    empty_tokens = []
    for token in tree_tokens:
        deps = token["deps"].split("|")
        for i, d in enumerate(deps):
            if ">" in d:
                # {head}:{empty_node_relation}>{current_node_relation}
                # should map to
                # For new, empty node:
                # {head}:{empty_node_relation}
                # For current node:
                # {new_empty_node_id}:{current_node_relation}
                # TODO consider where to put new_empty_node_id (currently at the end)
                head, relation = d.split(':', 1)
                ehead = f"{len(tree_tokens)}.{len(empty_tokens) + 1}"
                empty_node_relation, current_node_relation = relation.split(">", 1)
                deps[i] = f"{ehead}:{current_node_relation}"
                empty_tokens.append(
                    {
                        "id": ehead,
                        "deps": f"{head}:{empty_node_relation}"
                    }
                )
        deps = sorted([d.split(":", 1) for d in deps], key=lambda x: float(x[0]))
        token["deps"] = "|".join([f"{k}:{v}" for k, v in deps])
    return empty_tokens
