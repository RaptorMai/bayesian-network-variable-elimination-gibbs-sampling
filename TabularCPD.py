import numpy as np
class TabularCPD():
    def __init__(self, variable, values, evidence=[]):
        self.cpd = self.create_cpd(values)
        self.target = variable
        self.parents = list(reversed(evidence))



    def create_cpd(self, prob):
        '''
        :param prob:
        :return: cpd table, ndarray
        '''
        '''
        create cpd, the order is for tmp is (target, parent(reversed order of evidence))
        for this example, [pollution, smoker]
        cpd_cancer = TabularCPD(variable='Cancer',
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],)
        '''
        prob_arr = np.array(prob)
        prob_fla = prob_arr.flatten("F")
        para_size = np.log2(prob_fla.size)
        tmp = []
        for i in range(int(para_size)):
            tmp.append(np.array([[True] * (2 ** i) + [False] * (2 ** i)] * (int(prob_fla.size) // (2 ** (i + 1)))).flatten())
        tmp.append(prob_fla)
        return np.column_stack(tmp)
