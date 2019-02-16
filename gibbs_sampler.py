import numpy as np
from Factor import Factor
from collections import defaultdict
import copy
from variable_elimination import VariableElimination as VE

class Gibbs_Sampler():
    def __init__(self, network):
        self.network = network
        return

    def query(self, targets, steps=2000, burn_in=200, plot_step=20, evidence={}):
        ret = {}
        for i in targets:
            val, detail = self.one_query(i, steps, burn_in, plot_step, evidence)
            ret[i] = [val, detail]
            self.network.reset_all()
        return ret

    def one_query(self, target, steps=2000, burn_in=200, plot_step=20, evidence={}):
        detailed = {'steps':[], '0_val':[], '1_val':[]}
        sample_dic = defaultdict(dict)
        state = {i: evidence[i] for i in evidence.keys()}
        sampling_nodes = set(self.network.nodes) - set(evidence.keys())
        ve = VE(self.network)
        for i in sampling_nodes:
            sample_dic[i][1] = 0
            sample_dic[i][0] = 0
            state[i] = np.random.choice([0, 1])
        for step in range(steps+burn_in):
            for node in sampling_nodes:
                copy_state = copy.deepcopy(state)
                del copy_state[node]
                ve.network.reset_all()
                sample_cpd = ve.one_query(node, copy_state)
                sample_prob = sample_cpd.prob[:, -1].flatten('F')
                if np.isnan(sample_prob).any():
                    ve.network.reset_all()
                    sample_cpd = ve.one_query(node, evidence)
                    sample_prob = sample_cpd.prob[:, -1].flatten('F')
                chosen_val = np.random.choice([0, 1], p=sample_prob)
                state[node] = chosen_val
            if step > burn_in:
                for node, val in state.items():
                    if node in sampling_nodes:
                        sample_dic[node][1-val] += 1
                if step % plot_step ==0:
                    detailed['steps'].append(step)
                    detailed['0_val'].append(round(int(sample_dic[target][0]) / float(step-burn_in),4))
                    detailed['1_val'].append(round(int(sample_dic[target][1]) / float(step-burn_in), 4))

        for node in sample_dic:
            sample_dic[node][0] = round(int(sample_dic[node][0]) / float(steps),4)
            sample_dic[node][1] = round(int(sample_dic[node][1]) / float(steps), 4)

        return sample_dic[target], detailed

    def create_factors(self, evidence):
        #new_factors_list = np.array([])
        new_factors_dict = {}
        for node in self.network.nodes:
            factor = self.make_factor(node, evidence)
            if factor:
                new_factors_dict[node] = factor
                #new_factors_list = np.append(new_factors_list, factor)
        return new_factors_dict #new_factors_list

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



