import numpy as np
from collections import defaultdict
'''
we find rank the order with min parents
'''
def find_order(network, evidece, query):
    parents_dic = defaultdict(int)
    for i in network.nodes:
        parents_dic[i] += len(network.parent[i])

    parents_dic.pop(query)
    if evidece:
        for i in evidece.keys():
            parents_dic.pop(i)
    num_parent = list(parents_dic.values())
    name = list(parents_dic.keys())
    order = np.argsort(num_parent)
    return np.array(name)[order]