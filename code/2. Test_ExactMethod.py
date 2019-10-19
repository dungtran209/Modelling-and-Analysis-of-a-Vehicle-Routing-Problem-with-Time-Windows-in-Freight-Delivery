#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import time

from scipy.spatial import distance_matrix

# In[14]:


import sys
import docplex.mp

# try:
#     import docplex.mp
# except:
#     if hasattr(sys, 'real_prefix'):
#         get_ipython().system('pip install docplex -q')
#     else:
#         get_ipython().system('pip install --user docplex -q')

from docplex.mp.model import Model

dir_name = os.path.dirname(os.path.realpath('__file__'))

# folder = ["R101", "R102", "R103", "R104", "R105", "R106", "R107", "R108", "R109", "R110", "R111", "R112",
#           "R201", "R202", "R203", "R204", "R205", "R206", "R207", "R208", "R209", "R210", "R211",
#           "RC101", "RC102", "RC103", "RC104", "RC105", "RC106", "RC107", "RC108",
#           "RC201", "RC202", "RC203", "RC204", "RC205", "RC206", "RC207", "RC208",
#           "C201", "C202", "C203", "C204", "C205", "C206", "C207", "C208",
#           "C101", "C102", "C103", "C104", "C105", "C106", "C107", "C108", "C109"]

file_names = ["R101", "R106", "C101", "C106", "RC101", "RC106", "R201", "R206", "C201", "C206", "RC201", "RC206"]

K = 10000

for file in file_names:
    file_name = os.path.join(dir_name, 'Sample Dataset\\', file + '.csv')
    df = pd.read_csv(file_name, encoding='latin1', error_bad_lines=False);

    df = df[0:26]

    cust_size = df.shape[0] - 1
    # print('Number of customers:', cust_size)
    df.head()

    n = cust_size
    Q = df['CAPACITY'][0]
    C = [i for i in range(1, n + 1)]
    Cc = [0] + C + [n + 1]
    V = [i for i in range(1, 26)]

    df2 = df.iloc[:, 1:3]
    df2.loc[n + 1, :] = df2.loc[0, :]

    dist_matrix = pd.DataFrame(distance_matrix(df2.values, df2.values), index=df2.index, columns=df2.index)

    # dist_matrix.head()

    time_start = time.time()
    mdl = Model('VRPTW')

    # Start time

    e = [df['READYTIME'][i] for i in range(n + 1)]
    e.append(df['READYTIME'][0])

    # Due time

    l = [df['DUETIME'][i] for i in range(n + 1)]
    l.append(df['DUETIME'][0])

    # Service time
    ser = [df['SERVICETIME'][i] for i in range(n + 1)]
    ser.append(df['SERVICETIME'][0])

    # Demand

    #r = {i: df['DEMAND'][i] for i in range(1, n + 1)}
    r = [df['DEMAND'][i] for i in range(n + 1)]
    r.append(0)

    # Variable set
    X = [(i, j, k) for i in Cc for j in Cc for k in V if i != j]
    S = [(i, k) for i in Cc for k in V]

    # Calculate distance and time
    c = {(i, j): dist_matrix[i][j] for i in Cc for j in Cc}
    t = {(i, j): dist_matrix[i][j] for i in Cc for j in Cc}

    # Variables
    x = mdl.binary_var_dict(X, name='x')
    s = mdl.continuous_var_dict(S, name='s')

    # Constraints
    mdl.sum(c[i, j] * x[i, j, k] for i, j, k in X)

    mdl.add_constraints(mdl.sum(x[i, j, k] for j in Cc for k in V if j != i) == 1 for i in C)

    mdl.add_constraints(mdl.sum(r[i] * mdl.sum(x[i, j, k]) for i in C for j in Cc if i != j) <= Q for k in V)

    mdl.add_constraints(mdl.sum(x[0, j, k] for j in Cc if j != 0) == 1 for k in V)

    mdl.add_constraints(
        (mdl.sum(x[i, p, k] for i in Cc if i != p) - mdl.sum(x[p, j, k] for j in Cc if p != j)) == 0 for p in C for k in
        V)

    mdl.add_constraints(mdl.sum(x[i, n + 1, k] for i in Cc if i != n + 1) == 1 for k in V)

    mdl.add_constraints(s[i, k] + ser[i] + t[i, j] - K * (1 - x[i, j, k]) - s[j, k] <= 0 for i, j, k in X if i != j)

    #mdl.add_constraints(s[i,k] + ser[i] + t[i,j] - (max(0, (l[i] + ser[i] + t[i,j] - e[j])))*(1-x[i,j,k]) - s[j,k] <=0 for i,j,k in X if i!=j if i!=n+1 if j!=0)

    mdl.add_constraints(s[0, k] == 0 for k in V)

    mdl.add_constraints(s[i, k] >= e[i] for i, k in S if i != 0)

    mdl.add_constraints(s[i, k] <= l[i] for i, k in S if i != 0)

    # Objective Function
    obj_function = mdl.sum(c[i, j] * x[i, j, k] for i, j, k in X)

    # Set time limit
    mdl.parameters.timelimit.set(1000)
    #mdl.parameters.emphasis.mip.set(3)
    #mdl.parameters.mip.tolerances.mipgap.set(0.4)
    #mdl.parameters.mip.strategy.probe.set(3)


    # Solve
    mdl.minimize(obj_function)

    time_solve = time.time()

    solution = mdl.solve(log_output = True)

    time_end = time.time()
    # print(solution)

    running_time = round(time_end - time_solve, 2)
    elapsed_time = round(time_end - time_start, 2)

    if solution != None:
        route = [x[0, i, k] for i in C for k in V if x[0, i, k].solution_value == 1]
        no_vehicles = len(route)
        obj = round(obj_function.solution_value, 2)
        print(file, cust_size, no_vehicles, obj, elapsed_time, running_time)
    else:
        print(file, cust_size, 'NA', 'NA', elapsed_time, running_time)

