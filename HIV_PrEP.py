# ################################################################################################ #
# #############                                                               #################### #
# #############       Infectious Simulation in Stochastic Blockmodel Networks #################### #
# #############                                                               #################### #
# ################################################################################################ #

import os, sys, random, cProfile, pstats, itertools, math
import collections, copy, time, pickle
import numpy as np, scipy as sy, networkx as nx, pandas as pd
from string import *
from operator import itemgetter
from scipy import stats as ss
np.set_printoptions(precision=3, suppress = True)

simulation_index = sys.argv[1]
simulation_index = int(simulation_index)
trial            = simulation_index - 1

data_filepath = "PrEP_Data"
if not os.path.exists(data_filepath):
    os.makedirs(data_filepath)

def logit(x):
    return np.log(x) - np.log(1-x)

def expit(x):
    return 1 / (1 + math.exp(-x))

def weighted_sample(choices, n):
    total   = sum(weight for value, weight in choices)
    randoms = sorted([random.uniform(0, total) for r in range(n)])
    choices = sorted(choices, key=itemgetter(1),reverse=True)
    sample  = []
    weight_sum = 0
    r = 0
    for choice, weight in choices:
        weight_sum += weight
        if r >= n:
            break
        if weight_sum > randoms[r]:
            sample.append(choice)
            r += 1
    return sample

effectivenesses = enumerate([.4, .55, .7, .85, 1.0])
effectiveness_granularity = 5
scenarios = 11
n         = 200
clusters  = 32
assortative_cluster = [True]*(clusters//2) + [True]*(clusters//2)
average_degree      = [3]   *(clusters//2) + [3]   *(clusters//2)
C         = 4
Lambda    = .1                 # community structure.
Mu        = .8                 # sex worker structure.
assortativity_threshold = 0.3  # +- 0.4 achievable with similar distributions by sex, which is close to the empirical limit (newman 2002).
M         = 5                  # number of epidemics to run.  Just in case, this is more than one.
ps        = [.3, .1]
IV_ps     = np.array([.30, .25])*0      # IV quantile is inherent here.  range from 0 to 1.  Not used in this simulation.
mu        = .30                         # probability of recovering.
recovered_infectibility_proportion = .1 # the proportion of original infectibility held by recovered.  IN SIR, this is 0.  in SI, this is 1.
quantile           = 25                 #+ random.choice(range(-4,4))
prop_seeds         = 10                 # 1, 10
baseline_quantile  = 25                 # 5, 25
study_end_quantile = 10                 # 30, currently not used.
baseline_total  = baseline_quantile /100 * n * clusters
study_end_total = study_end_quantile/100 * n * clusters
baseline_treated   = 0.55               # the findability parameter for epidemic start to baseline.

TasP_effectiveness = 0.90
PrEP_effectiveness = 0.90
TasP_tenacity      = 0.80
PrEP_tenacity      = 0.80

design = 1 # if design == 1, tasp and prep findability are constant throughout the study.
           # if design == 2, tasp and prep are set at 85% at year 3.

verbose      = False
timeout      = 50
study_end    = 7
TasPs_random = random.sample(range(n), int(n))
PrEPs_random = random.sample(range(n), int(n))
base_random  = random.sample(range(n), int(n))

total_data     = pd.DataFrame({})
graphs         = [nx.Graph()]*clusters
infs           = [{}]*clusters # infectibility.
suss           = [{}]*clusters # susceptibility.  probability of infection is inf[infected] * suss[susceptible].
ccs            = [{}]*clusters # concurrency per node.  Currently unexamined.
iis            = [{}]*clusters # infectivity per node.
LCCs           = [{}]*clusters # size of the largest connected components.
seedss         = [{}]*clusters # number of seeds in each cluster.
degreess       = [{}]*clusters # degrees per node in each cluster.
infecteds      = [{}]*clusters # infected nodes in each cluster.
recovereds     = [{}]*clusters # recovered nodes in each cluster.
controllers    = [{}]*clusters # controllers in each cluster.
TasPs_findable = [{}]*clusters # findable nodes for TasP in each cluster.
PrEPs_findable = [{}]*clusters # findable nodes for PrEP in each cluster.
base_findable  = [{}]*clusters # treated during baseline run.
ts             = [0]*M

# Generate a bipartite degree-corrected stochastic blockmodel with assortative rewiring (preserving degree and block).
for cluster in range(clusters):
    while len(LCCs[cluster]) < study_end_quantile/100* n:  # this ensures that every graph has a LCC that can sustain a sufficiently-sized epidemic.
        assortative = True # assortative_cluster[cluster] # assumes assortativity for all clusters.
        infectivity = ["degree", "degree"]
        concurrency = ["degree","degree"]
        k_female =  ((average_degree[cluster]-1)*ss.uniform().rvs(n/2)+1).astype(int)*ss.zipf(2.5).rvs(n/2) # will require some editing if mean(degree) differs from K.
        k_male   =  ((average_degree[cluster]-1)*ss.uniform().rvs(n/2)+1).astype(int)*ss.zipf(2.5).rvs(n/2)
##        k_female = ss.poisson(average_degree[cluster]).rvs(n/2)
##        k_male   = ss.poisson(average_degree[cluster]).rvs(n/2)

        counter_threshold = 30
        k = np.concatenate((k_female, k_male)) # this assumes both bipartite halves they have the same distribution.
        k[k>n] = n                             # eliminates impossibly high values.
        g = {i: 2*C*i//n for i in range(n)}
        kappa = [np.sum([k[i] for i in range(n) if g[i] == K]) for K in range(2*C)]
        m = sum(kappa)/2
        theta = [k[i] / kappa[g[i]] for i in range(n)]
        omega_random = np.zeros((2*C,2*C))
        omega_zeros  = np.zeros((C,C))
        omega_block  = np.zeros((C,C))
        for i in range(C):
            for j in range(C):
                omega_block[i,j] = kappa[i]*kappa[j] / (2*m)
            omega_random[:C,C:(2*C)] = omega_block
            omega_random[C:(2*C),:C] = omega_block

        omega_planted = np.zeros((2*C,2*C))
        for i in range(2*C):
            for j in range(2*C):
                if i == ((j+C) % (2*C)):
                    omega_planted[i,j] = kappa[i]/2

        omega_sexwork       = copy.deepcopy(omega_planted)
        omega_sexwork[:C,C] = copy.deepcopy(omega_random[:C,C])
        omega_sexwork[C,:C] = copy.deepcopy(omega_random[C,:C])
        omega_sexwork[C:,0] = copy.deepcopy(omega_random[C:,0])
        omega_sexwork[0,C:] = copy.deepcopy(omega_random[0,C:])

        omega = Lambda * omega_planted + Mu * omega_sexwork + (1 - Lambda - Mu) * omega_random

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(2*C):
            for j in range(2*C):
                mean_edges = int(omega[i,j])
                if i == j:
                    mean_edges /= 2
                pois    = np.random.poisson(mean_edges)
                ns_i    = [node for node in range(n) if g[node] == i]
                ns_j    = [node for node in range(n) if g[node] == j]
                p_i     = [theta[node] for node in ns_i]
                p_j     = [theta[node] for node in ns_j]
                stubs_i = [np.random.choice(ns_i,p=p_i) for stub in range(pois)]
                stubs_j = [np.random.choice(ns_j,p=p_j) for stub in range(pois)]
                graph.add_edges_from(zip(stubs_i, stubs_j))
        LCCs[cluster] = sorted(nx.connected_components(graph), key = len, reverse = True)[0]

        degrees       = list(graph.degree().values())
        m             = len(graph.edges())
        counter       = 0
        assortativity = 0
        edges = graph.edges()
        omega_edges = [[[edge for edge in edges if ((g[edge[0]] == i and g[edge[1]] == j) or (g[edge[0]] == j and g[edge[1]] == i))] for j in range(2*C)] for i in range(2*C)] # a matrix of edge lists according to 2*C membership.  Usage: omega_edges[i][j] is a bunch of edges.

        while assortativity_threshold - ((2*int(assortative)-1) * assortativity) > 0 and counter < counter_threshold:
            for i in range(100):
                random_edge = random.choice(graph.edges())
                same_block_edge = random.choice(omega_edges[g[random_edge[0]]][g[random_edge[1]]]) # select another edge from the same omega block.
                current_edges = (random_edge, same_block_edge)
                remaining_degrees = [[degrees[stub] for stub in edge] for edge in current_edges]
                current_degree_difference  = abs(remaining_degrees[0][0] - remaining_degrees[0][1]) + abs(remaining_degrees[1][0] - remaining_degrees[1][1])
                rewiring_degree_difference = abs(remaining_degrees[0][0] - remaining_degrees[1][1]) + abs(remaining_degrees[0][1] - remaining_degrees[1][0])
                if (2*int(assortative)-1) * (current_degree_difference - rewiring_degree_difference) > 0 :
                    new_edges = ((random_edge[0],same_block_edge[1]),(random_edge[1],same_block_edge[0]))
                    graph.remove_edges_from(current_edges)
                    graph.add_edges_from(new_edges)
            assortativity = nx.degree_pearson_correlation_coefficient(graph)
            counter += 1
    graphs[cluster] = graph

# end network generation.  start the epidemic.
for cluster in range(clusters):
    seedss[cluster] = random.sample(graphs[cluster].nodes(), int(max(float(n)*prop_seeds/100,1))) #LCCs[cluster]
    degreess[cluster] = list(graphs[cluster].degree().values())
    infecteds[cluster] = set(seedss[cluster])
    recovereds[cluster] = set()
    base_findable[cluster] = set()
    ccs[cluster] = {node: (graphs[cluster].degree(node) - 1)*(int(concurrency[node//n] == "degree")) + 1 for node in range(n)}
    iis[cluster] = {node: (graphs[cluster].degree(node) - 1)*(int(infectivity[node//n] == "degree")) + 1 for node in range(n)}
    infs[cluster] = {node: ps[0]*float(iis[cluster][node])/(ccs[cluster][node] + int(ccs[cluster][node]==0)) for node in range(n)} # everyone is untreated to begin.
    controllers[cluster] = random.sample(range(n), n//10) # select 10% of the population to be controllers.
    for controller in controllers[cluster]:
        infs[cluster][controller] /= 10 # reduce the susceptibility of controllers.
        base_findable[cluster] = random.sample(range(n), int(n*baseline_treated))
    suss[cluster] = {node: 1 for node in range(n)}

    t = 0
    if cluster == 0:   # printing statistics for only one cluster.
        print("trial " + str(trial))
        print("proportion of target degree:" + str(float(sum(degrees)) / sum(k)))
        print("assort: " + str(nx.degree_pearson_correlation_coefficient(graph)))
        print(assortativity, counter)
        print(np.mean(list(graph.degree().values())), np.mean(k))
        print(float(len(seedss[cluster]))/n)

current_prevalence = sum(len(infecteds[cluster]) for cluster in range(clusters))
class Baseline_Reached(BaseException): pass
try:
    for t in range(timeout):
        for cluster in random.sample(range(clusters), clusters):
            recoverable = infecteds[cluster].difference(recovereds[cluster])
            for infected in recoverable:
                if random.random() < mu:
                    recovereds[cluster].add(infected)
            if t > 5:
                for base_found in base_findable[cluster]:
                    infs[cluster][base_found] = (1-TasP_effectiveness) * ps[0]*float(iis[cluster][base_found])/(ccs[cluster][base_found] + int(ccs[cluster][base_found]==0))
            newly_infected = set()
            for infected_node in random.sample(infecteds[cluster],len(infecteds[cluster])):
                neighbors = graphs[cluster].neighbors(infected_node)
                if len(neighbors) > 0:
                    for neighbor in random.sample(neighbors, ccs[cluster][infected_node]):
                        probability = infs[cluster][infected_node] * suss[cluster][neighbor]
                        if infected_node in recovereds[cluster]:
                            probability *= recovered_infectibility_proportion
                        successful_infection = random.random() < probability
                        if successful_infection and neighbor not in infecteds[cluster]:
                            current_prevalence += 1
                            newly_infected.add(neighbor)
                            infecteds[cluster] = infecteds[cluster].union(newly_infected)
                            if current_prevalence >= baseline_total:
                                raise Baseline_Reached()
except Baseline_Reached:
    pass

prevalences        = copy.deepcopy(infecteds)
initial_infs       = copy.deepcopy(infs)
initial_suss       = copy.deepcopy(suss)
initial_recovereds = copy.deepcopy(recovereds)

# set up each scenario.
datas = [{}]*clusters
prevalence_outcomes = [{}]*clusters
for cluster in range(clusters):
    prevalence_outcomes[cluster] = copy.deepcopy({node: 0 for node in graphs[cluster].nodes()})
    for infected_node in prevalences[cluster]:
        prevalence_outcomes[cluster][infected_node] = 1
    IV_weights  = [{node: np.random.normal(logit(IV_ps[2*cluster//clusters]), 1, 1)[0]  for node in range(n)} for cluster in range(clusters)]
    IV_nodes    = set.union(*[{((cluster, node), expit(IV_weights[cluster][node])) for node in set(range(n)).difference(infecteds[cluster])} for cluster in range(clusters)])
    IV_infected = set([])
    for IV_node in IV_nodes:
        if random.random() < IV_node[1]:
            IV_infected.add(IV_node[0])

    mins = {node: 0 for node in range(n)}
    sums = {node: 0 for node in range(n)}
    for node in range(n):
        paths = [nx.shortest_path_length(graphs[cluster],node,neighbor) for neighbor in set(nx.node_connected_component(graphs[cluster], node)).intersection(prevalences[cluster]).difference(set([node]))]
        if len(paths) == 0:
            mins[node] = 0
            sums[node] = 0
        if len(paths) > 0:
            mins[node] = 1 / min(paths)
            sums[node] = sum([1/path for path in paths])

    components = [len(C) for C in nx.connected_components(graphs[cluster])]
    datas[cluster] = pd.DataFrame({
        # main informations relevant to basic analysis
        "Trt":                  (2*cluster//clusters),
        "Cluster":              cluster,
        "Prevalences":          prevalence_outcomes[cluster],

        # Degree-based covariates
        "Degree":               graphs[cluster].degree(),
        "Mean_Neighbor_Degree": nx.average_neighbor_degree(graphs[cluster]),
        "Assortativity":        nx.degree_assortativity_coefficient(graphs[cluster]),

        # Community-based covariates
        "Sex_Work":           {node: node < n/(C*2) or (node>(n//2) and node < (n//2 +n/(C*2))) for node in range(n)},

        # Component-based covariates
        "LCC_Size":             max(components),
        "Mean_Component_Size":  np.mean(components),
        "Number_Of_Components": len(components),
        "Node_Component_Size":  {node: len(nx.node_connected_component(graphs[cluster], node)) for node in range(n)},

        # Infectious path-based covariates
        "Total_Neighbor_Seeds": {node: len(set(graphs[cluster].neighbors(node)).intersection(set(prevalences[cluster]))) for node in range(n)},
        "Total_Cluster_Seeds":  {node: len(set(nx.node_connected_component(graphs[cluster], node)).intersection(prevalences[cluster])) for node in range(n)},
        "Mins":                 mins, # inverse of the minimum path length to an infected node at baseline.
        "Sums":                 sums, # sum of the inverse path length to an infected node at baseline.

        # Ancillary network features not yet being utilized
        "Age_Group":            {node: g[node]%C for node in g},
        "Sex":                  {node: g[node]//C for node in g},
        "IV_weights":           IV_weights[cluster]
    })

for scenario in range(scenarios):
    for finding_effectiveness_number, finding_effectiveness in copy.deepcopy(effectivenesses):
        infecteds   = copy.deepcopy(prevalences)
        infs        = copy.deepcopy(initial_infs)
        suss        = copy.deepcopy(initial_suss)
        recovereds  = copy.deepcopy(initial_recovereds)
        PrEPs_found = [0]*clusters # this dictionary is > 0 for those found IF findable.  You need the intersection of both.
        TasPs_found = [0]*clusters # same for TasP.
        PrEPs_PY    = [0]*clusters # the person-years on PrEP, if any.
        TasPs_PY    = [0]*clusters # same for TasP.
        couples     = [0]*clusters # the number of couples.
        outcomes    = [0]*clusters # this dictionary is the PY infected.

        for cluster in range(clusters):
            TasPs_findable[cluster] = TasPs_random[:int(finding_effectiveness*n)] # The complete list of individuals that could potentially be found for TasP.
            PrEPs_findable[cluster] = PrEPs_random[:int(finding_effectiveness*n)] # Same for PrEP.
            TasPs_found[cluster]    = set()
            PrEPs_found[cluster]    = set()
            TasPs_PY[cluster]       = {node: 0 for node in range(n)}
            PrEPs_PY[cluster]       = {node: 0 for node in range(n)}
            outcomes[cluster]       = {node: 0 for node in range(n)}
            couples[cluster]        = set()
            for node in range(n):
                neighbors = set(graphs[cluster].neighbors(node)).difference(prevalences[cluster])
                if len(neighbors) > 0:
                    couples[cluster].add(random.choice(list(neighbors)))
        current_incidence = sum(len(infecteds[cluster]) for cluster in range(clusters))

        class End_Reached(BaseException): pass
        try:
            for t in range(study_end):
                for cluster in random.sample(range(clusters), clusters):
                    if t == 3:
                        datas[cluster]["Sum_Neighbors_Infected_"  + str(scenario) + "_" + str(finding_effectiveness_number)] = np.array([len(set(graphs[cluster].neighbors(node)).intersection(infecteds[cluster])) for node in graphs[cluster].nodes()])
                    if design == 2 and t > 2:
                        TasPs_findable[cluster] = TasPs_random[:int(.85*n)] # this changes the findability at year 2, if design 2.
                        PrEPs_findable[cluster] = PrEPs_random[:int(.85*n)]
                    for node in range(n):
                        if scenario != 0 and (scenario in range(6) or node not in controllers[cluster]):# if scenario 0 (null), don't do any of this.  If scenario in 1-5, treat all findable TasPs.  Otherwise, TasP only non-controlling findable TasPs.
                            if node in TasPs_findable[cluster] and random.random() < TasP_tenacity:
                                TasPs_found[cluster].add(node)
                    if scenario in (3,3+5) and t > 2:
                        for node in set(range(n//(C*2))).union(set(range(n//2,n//2+n//(C*2)))).intersection(PrEPs_findable[cluster]):
                            if random.random() < PrEP_tenacity:
                                PrEPs_found[cluster].add(node)
                    if scenario in (4,4+5) and t > 2:
                        for node in set([node for node in range(n) if len(graphs[cluster][node]) > 3]).intersection(PrEPs_findable[cluster]):
                            if random.random() < PrEP_tenacity:
                                PrEPs_found[cluster].add(node)
                    if scenario in (5,5+5) and t > 2:
                        for node in couples[cluster].intersection(PrEPs_findable[cluster]):
                            if random.random() < PrEP_tenacity:
                                PrEPs_found[cluster].add(node)

                    for node in PrEPs_found[cluster]:
                        PrEPs_PY[cluster][node] += 1
                        suss[cluster][node] = (1-PrEP_effectiveness)
                    for node in TasPs_found[cluster]:
                        infs[cluster][node] = (1 - TasP_effectiveness) * ps[0] * float(iis[cluster][node])/(ccs[cluster][node] + int(ccs[cluster][node]==0))#ps[(2*cluster//clusters)] does a CRT
                        TasPs_PY[cluster][node] += 1

                    recoverable = infecteds[cluster].difference(recovereds[cluster])
                    for infected in recoverable:
                        if random.random() < mu:
                            recovereds[cluster].add(infected)
                    newly_infected = set()
                    for infected_node in random.sample(infecteds[cluster],len(infecteds[cluster])):
                        neighbors = graphs[cluster].neighbors(infected_node)
                        if len(neighbors) > 0:
                            for neighbor in random.sample(neighbors, ccs[cluster][infected_node]):
                                probability = infs[cluster][infected_node] * suss[cluster][neighbor]
                                if infected_node in recovereds[cluster]:
                                    probability *= recovered_infectibility_proportion
                                successful_infection = random.random() < probability
                                if successful_infection and neighbor not in infecteds[cluster]:
                                    current_incidence += 1
                                    newly_infected.add(neighbor)
                                    infecteds[cluster] = infecteds[cluster].union(newly_infected)
                                    if scenario in (2,2+5) and t > 2:
                                        for surround1 in graphs[cluster].neighbors(neighbor):
                                            for surround2 in graphs[cluster].neighbors(surround1):
                                                if surround2 in PrEPs_findable[cluster] and random.random() < PrEP_tenacity:
                                                    PrEPs_found[cluster].add(node)
                                    if t > study_end: # current_incidence >= study_end_total: for time-ended CRT.  If stopped at time, just counterfactuals after a fixed stopping time!
                                        raise End_Reached()
                    for node in infecteds[cluster]:
                        outcomes[cluster][node] += 1

        except End_Reached:
            pass
        if trial == 0:
            print(scenario, "\t", finding_effectiveness, "\t", np.round(np.mean(list(TasPs_PY[cluster].values())),3), "\t", np.round(np.mean(list(PrEPs_PY[cluster].values())),3), "\t", np.round(np.mean([len(infecteds[cluster]) for cluster in range(clusters)]),3))
        # store the person-years of how long they were prepped and infected.
        for cluster in range(clusters):
            datas[cluster]["outcome_" + str(scenario) + "_" + str(finding_effectiveness_number)] = np.array(list(outcomes[cluster].values()))
            datas[cluster]["prepped_" + str(scenario) + "_" + str(finding_effectiveness_number)] = np.array(list(PrEPs_PY[cluster].values()))
            datas[cluster]["tasped_"  + str(scenario) + "_" + str(finding_effectiveness_number)] = np.array(list(TasPs_PY[cluster].values()))

for cluster in range(clusters): # rearrange the data for easy analysis.
    datas[cluster] = datas[cluster][["outcome_" + str(s) + "_" + str(f) for s in range(scenarios) for f in range(effectiveness_granularity)]+
    ["prepped_" + str(s) + "_" + str(f) for s in range(scenarios) for f in range(effectiveness_granularity)]+
    ["tasped_"  + str(s) + "_" + str(f) for s in range(scenarios) for f in range(effectiveness_granularity)]+
    ["Sum_Neighbors_Infected_"  + str(s) + "_" + str(f) for s in range(scenarios) for f in range(effectiveness_granularity)]+[
        "Cluster", "Prevalences",
        "Degree", "Mean_Neighbor_Degree", "Assortativity",
        "Sex_Work",
        "LCC_Size", "Mean_Component_Size", "Number_Of_Components", "Node_Component_Size",
        "Total_Neighbor_Seeds", "Total_Cluster_Seeds", "Mins", "Sums",
        "Age_Group", "Sex", "IV_weights"
        ]]
total_data = pd.concat(datas)
total_data.to_csv(data_filepath + "/" + "CRT_"+str(trial)+".txt", sep = "\t", index = False)

tot = [len(i) for i in infecteds]
beta = logit(np.mean(tot[(clusters//2):])/n) - logit(np.mean(tot[:(clusters//2)])/n)
print(beta)

print(omega)
for i in omega_edges:
    print([len(j) for j in i])





























