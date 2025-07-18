from __future__ import annotations
from typing import Union, Sequence


class Node:
    '''
    Node type for the assembly tree
    '''

    def __init__(self, part_idx: int = None, children: Sequence[Node] = None, **kwargs):
        if part_idx is not None:
            assert children is None
            self.part_idxs = frozenset([part_idx])
            self.children = []
        else:
            assert part_idx is None and len(children) > 0
            self.children = children
            s = set()
            for c in self.children:
                s = s.union(c.part_idxs)
            self.part_idxs = frozenset(s)

        for k, v in kwargs:
            setattr(self, k, v)

    def __str__(self, level=0):

        ret = '\t' * level + f'Node(part_idxs={list(self.part_idxs)})' + '\n'
        for child in self.children:
            ret += child.__str__(level + 1)

        return ret


def build_tree_from_list(l):
    if isinstance(l, int):
        return Node(part_idx=l)
    return Node(children=[build_tree_from_list(n) for n in l])


