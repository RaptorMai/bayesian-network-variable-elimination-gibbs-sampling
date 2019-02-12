import numpy as np

class Factor:
    def __init__(self, nodes, prob, network):
        '''

        :param nodes: ndarray
        :param prob:
        :param network:
        '''
        self.nodes = nodes
        self.prob = prob
        self.network = network
        return

    def __str__(self):
        string = "Nodes: " + str(self.nodes) + "\n"
        string += "Probabilities:\n" + str(self.prob) + "\n"
        return string
        # ret = np.array([])
        # ret = np.append(ret, [str(self.nodes[0]), 'phi('+str(self.nodes[0])+')'])
        # ret = np.append(ret, str(self.prob))
        # return str(ret)

    def reduce(self, evidence):
        ordered_evidence_list = self.nodes[np.in1d(self.nodes, list(evidence.keys()))]
        self.reorder(ordered_evidence_list)
        # after this, in self.prob, all observed will be on the left-hand

        evidence_value_list = [1-evidence[i] for i in ordered_evidence_list]
        row_num = self.prob.shape[0]
        selected = [0, row_num]
        # select which row to pick
        for val in evidence_value_list:
            row_num = row_num//2
            selected[0] += val * row_num
            selected[1] -= (1 - val) * row_num
        unobserved_index = len(ordered_evidence_list)
        prob = self.prob[selected[0]:selected[1], unobserved_index:]

        return Factor(self.nodes[unobserved_index:], prob, self.network)

    def reorder(self, observed_nodes):
        #reorder the self.nodes, and self.probs, so that all observed are at the front
        for i, node in enumerate(observed_nodes):
            #print(self.nodes, node)
            index = np.flatnonzero(self.nodes == node)[0]
            if i != index:
                self.nodes[[i, index]] = self.nodes[[index, i]]
                self.prob[:, [i, index]] = self.prob[:, [index, i]]

        # sorted self.prob by the observed order
        dtype = [(str(i), bool) for i in range(self.nodes.size)] + [('prob', np.float)]
        self.prob = np.array(list(map(tuple, self.prob)), dtype=dtype)
        self.prob = np.sort(self.prob, order=list(map(str, range(self.nodes.size))))
        self.prob = np.array(list(map(list, self.prob)))

    def times(self, factor):
        #how many common nodes
        common_nodes = np.intersect1d(self.nodes, factor.nodes)
        num_common_nodes = common_nodes.size
        # num_common_nodes_total_combinations is the total #of combination of the commom nodes
        num_common_nodes_total_combinations = 2**(num_common_nodes)

        #all the common nodes are on the left sorted, self.nodes as well
        self.reorder(common_nodes)
        factor.reorder(common_nodes)

        # nodes have all the nodes of this product
        all_nodes = np.append(self.nodes, factor.nodes[num_common_nodes:])

        #product1 and product2 has the same number of row but different number of coloumn
        #turn the factor into a factor that have the same # rows, each row have the same prob of the common nodes

        new_factor_self = self.rebuild_factor(num_common_nodes_total_combinations)
        new_another_factor = factor.rebuild_factor(num_common_nodes_total_combinations)


        # new_factor_self[i, e] every number * new_another_factor[i], the product has size [#product1_col x # product2_col] x num_common_nodes_total_combinations
        products_list = [[new_factor_self[i, k] * new_another_factor[i] for k in range(new_factor_self.shape[1])]
                         for i in range(num_common_nodes_total_combinations)]

        #every ele in products+cp;umns is a single-col ndarray, so vstack piles all the 1-col into a big 1-col
        products_columns = [product.reshape((-1, 1)) for products in products_list for product in products]
        products = np.vstack(products_columns)

        probs = Factor.gen_prob_table(self, factor, products, num_common_nodes)

        return Factor(all_nodes, probs, self.network)

    def rebuild_factor(self, num_common_nodes_total_combinations):
        nr_equal_common_nodes_values = self.prob.shape[0] // num_common_nodes_total_combinations
        bounds_list = [[i * nr_equal_common_nodes_values, (i + 1) * nr_equal_common_nodes_values]
                       for i in range(num_common_nodes_total_combinations)]
        products = [self.prob[bounds[0]:bounds[1], -1] for bounds in bounds_list]
        return np.array(products).astype(np.float)


    def marginalization(self, node):
        '''

        :param node: the node we want to marginalize
        :return: new factor after marginalize node
        '''

        #put node to the most left of self.probs and self.nodes
        self.reorder([node])
        # because node is binary, half of rows True, half False
        half_rows = self.prob.shape[0] // 2

        #we don't need the first col, we marginalize, we don't need the prob col, we make new prob
        new_nodes = self.prob[:half_rows, 1:-1]

        #inicdes list size #half_rows * 2
        indices_list = [[k * half_rows + i for k in range(2)]
                        for i in range(half_rows)]
        sums_list = [self.prob[indices, -1] for indices in indices_list]
        sums = np.array(sums_list).astype(np.float)
        sums = np.sum(sums, axis = 1).reshape((-1, 1))

        probs = np.hstack((new_nodes, sums))

        return Factor(self.nodes[1:], probs, self.network)


    def normalize(self, query):

        # reorder the self.prob and self.nodes by putting all the query to the most left
        self.reorder([query])

        new_nodes = self.prob[:, :-1].reshape((-1, self.nodes.shape[0]))
        prob_column = self.prob[:,-1].astype(np.float).reshape((-1, 1))
        prob_column = prob_column / np.sum(prob_column)
        prob_column = prob_column[::-1]
        probs = np.hstack((new_nodes, prob_column))

        return Factor(self.nodes, probs, self.network)

    @staticmethod
    def gen_prob_table(factor1, factor2, prob_column, num_common_nodes):
        '''
        computer the final prob table

        :param factor1: factors on the most left
        :param factor2: factors after factor1
        :param prob_column: prob values
        :param num_common_nodes: obvious
        :return: final prob table
        '''

        prob1 = factor1.prob[:, :-1]  # Includes common nodes columns
        prob2 = factor2.prob[:, num_common_nodes:-1]  # Excludes common nodes columns


        #both factor1 and factor2 have more nodes other than common nodes
        if factor1.prob.shape[0] > num_common_nodes and factor2.prob.shape[0] > num_common_nodes:
            nr_rows = prob_column.size
            #difference between repeat and tile is, repeat repeat ele one by one, tile repeat the whole thing together
            probs = [np.repeat(prob1, nr_rows // factor1.prob.shape[0] , axis=0)]
            probs += [np.tile(prob2.T, nr_rows // factor2.prob.shape[0]).T]

        #only factor2 has more nodes other than common nodes
        elif factor2.prob.shape[0]  > num_common_nodes:
            # if only factor2 has more rows, then #row of probs == factor2 #rows
            probs = [np.repeat(prob1, factor2.nr_rows // factor1.prob.shape[0] , axis=0)]
            probs += [prob2]

        #only factor1 has more nodes then
        else:
            probs = [prob1]
        probs += [prob_column]

        return np.hstack(probs)

