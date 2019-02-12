import numpy as np
from Factor import Factor
from find_order import find_order

class VariableElimination():
    def __init__(self, network):
        self.network = network
        self.multiplication = 0
        return

    def query(self, targets, evidence=None):
        ret = {}
        for i in targets:
            ret[i] = self.one_query(i, evidence)
            self.network.reset_all()
        return ret

    def one_query(self, target, evidence=None):
        factors_list = self.create_factors(evidence)
        order = find_order(self.network, evidence, target)
        #print(factors_list)
        for variable in order:
            # all factors that contains variable
            factors_elim_indices = [idx for idx, factor in enumerate(factors_list) if variable in factor.nodes]
            factors_elim = factors_list[factors_elim_indices]
            length = factors_elim.size
            if length > 0:
                self.multiplication += length - 1
                new_factor = self.factor_multiplication(length, factors_elim.copy())
                if new_factor.nodes.size > 1:
                    marginalization = new_factor.marginalization(variable)
                    factors_list = np.append(factors_list, marginalization)
            factors_list = np.delete(factors_list, factors_elim_indices)


        result = self.factor_multiplication(factors_list.size, factors_list.copy())
        self.multiplication += factors_list.size
        result = result.normalize(target)
        return result

    def factor_multiplication(self, length, dummy_factors_elim):
        for i in range(length-1):
            dummy_factors_elim[1] = dummy_factors_elim[0].times(dummy_factors_elim[1])
            dummy_factors_elim = np.delete(dummy_factors_elim, 0)
        return dummy_factors_elim[0]

    def create_factors(self, evidence):
        new_factors_list = np.array([])
        for node in self.network.nodes:
            factor = self.make_factor(node, evidence)
            if factor:
                new_factors_list = np.append(new_factors_list, factor)
        return new_factors_list

    def make_factor(self, node, evidence):
        # the order is fixed, that's the order for prob

        nodes = np.array([node] + self.network.CPDs[node].parents)
        prob = self.network.CPDs[node].cpd
        if not evidence:
            nodes_observed = np.array([])
        else:
            nodes_observed = nodes[np.in1d(nodes, list(evidence.keys()))]
        if nodes_observed.size > 0:
            # only some parents or the node are observed
            if nodes.size > nodes_observed.size:
                #reduce the rows in factors that are observed
                tmp = Factor(nodes, prob, self.network).reduce(evidence)
                return tmp

            else:
                # if both the nodes and its parents are observed, the ancestors and children are independent
                return None
        else:
            # if both the node and its parents are not observed, create a factor
            return Factor(nodes, prob, self.network)