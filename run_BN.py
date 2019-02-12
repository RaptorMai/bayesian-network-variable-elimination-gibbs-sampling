from BayesNet import BayesNet
from TabularCPD import TabularCPD
from variable_elimination import VariableElimination

a = [('Pollution', 'Cancer'),
      ('Smoker', 'Cancer'),
      ('Cancer', 'Xray'),
      ('Cancer', 'Dyspnoea')]

BN = BayesNet(a)

cpd_poll = TabularCPD(variable='Pollution',
                      values=[[0.9], [0.1]])
cpd_smoke = TabularCPD(variable='Smoker',
                       values=[[0.3], [0.7]])
cpd_cancer = TabularCPD(variable='Cancer',
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],)
cpd_xray = TabularCPD(variable='Xray',
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'])
cpd_dysp = TabularCPD(variable='Dyspnoea',
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'])
BN.add_cpds([cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp])
infer = VariableElimination(BN)
a = infer.query(['Xray'], {'Smoker': 0, 'Pollution':0})
print(a['Xray'])