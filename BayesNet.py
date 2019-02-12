import copy

class BayesNet():
    def __init__(self, list_edges):
        self.nodes = list(set([k for i in list_edges for k in i]))
        self.CPDs = {}
        self.parent = {}
        self.reset = {'node': self.nodes}


    def add_cpds(self, cpds):
        self.reset['cpds'] = cpds
        self.add_cpds_real(copy.deepcopy(self.reset['cpds']))

    def add_cpds_real(self, cpds):
        for cpd in cpds:
            self.CPDs[cpd.target] = cpd
            self.parent[cpd.target] = cpd.parents

    def reset_all(self):
        self.CPDs = {}
        self.parent = {}
        self.nodes = copy.deepcopy(self.reset['node'])
        self.add_cpds_real(copy.deepcopy(self.reset['cpds']))
