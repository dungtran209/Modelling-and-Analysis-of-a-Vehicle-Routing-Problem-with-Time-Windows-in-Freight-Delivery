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


file_names = ["9Aug19"]
K = 100000

for file in file_names:
    file_name = os.path.join(dir_name, 'Real Dataset\\', file + '.csv')
    df = pd.read_csv(file_name, encoding='latin1', error_bad_lines=False)

    d = {'Name': 'first', 'Post Code': 'first', 'Date': 'first', 'Reference': 'first', 'Type': 'first', 'Value': 'sum'}
    df_new = df.groupby('No', as_index=False).aggregate(d).reindex(columns=df.columns)


    cust_size = df_new.shape[0] - 1
    # print('Number of customers:', cust_size)
    n = cust_size
    print(n)
    Q = 4000
    C = [i for i in range(1, n + 1)]
    print(C)
    Cc = [0] + C + [n + 1]
    print(Cc)
    V = [i for i in range(1, cust_size+1)]
    print(V)

    dist_matrix = pd.read_csv('Real Dataset\\9AugDist.csv', encoding='latin1', error_bad_lines=False, header=None)
    #dist_matrix = dist_matrix_raw.transpose()
    dist_matrix.loc[n+1,:] = dist_matrix.loc[0,:]
    dist_matrix.loc[:,n+1] = dist_matrix.loc[:,0]
    print(dist_matrix)
    print(dist_matrix[0][1])


    time_matrix = pd.read_csv('Real Dataset\\9AugTime.csv', encoding='latin1', error_bad_lines=False, header=None)
    #time_matrix = time_matrix_raw.transpose()
    time_matrix.loc[n+1,:] = time_matrix.loc[0,:]
    time_matrix.loc[:,n+1] = time_matrix.loc[:,0]
    print(time_matrix)

    # dist_matrix.head()

    time_start = time.time()
    mdl = Model('VRPTW')

    # Start time

    e = [0] * (cust_size+2)
    print(e)

    # Due time

    l = [21600] * (cust_size+1)
    l[0] = 25200
    l.append(25200)
    print(l)


    # Demand

    #r = {i: df_new["Value"][i] for i in range(1, n + 1)}
    r = [df_new["Value"][i] for i in range(n + 1)]
    r.append(0)
    print(r)

    # Service time
    ser = [548 + 0.52*df_new["Value"][i] for i in range(n + 1)]
    ser[0] = 0
    ser.append(0)
    print(ser)

    # Variable set
    X = [(i, j, k) for i in Cc for j in Cc for k in V if i != j]
    S = [(i, k) for i in Cc for k in V]

    # Calculate distance and time
    c = {(i, j): dist_matrix[i][j] for i in Cc for j in Cc}
    t = {(i, j): time_matrix[i][j] for i in Cc for j in Cc}

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

    #mdl.add_constraints(s[i, k] + ser[i] + t[i, j] - K * (1 - x[i, j, k]) - s[j, k] <= 0 for i, j, k in X if i != j)

    mdl.add_constraints(s[i,k] + ser[i] + t[i,j] - (max(0, (l[i] + ser[i] + t[i,j] - e[j])))*(1-x[i,j,k]) - s[j,k] <=0 for i,j,k in X if i!=j if i!=n+1 if j!=0)

    mdl.add_constraints(s[0, k] == 0 for k in V)

    mdl.add_constraints(s[i, k] >= e[i] for i, k in S if i != 0)

    mdl.add_constraints(s[i, k] <= l[i] for i, k in S if i != 0)

    # Objective Function
    obj_function = mdl.sum(c[i, j] * x[i, j, k] for i, j, k in X)

    # Set time limit
    mdl.parameters.timelimit.set(1000)

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


