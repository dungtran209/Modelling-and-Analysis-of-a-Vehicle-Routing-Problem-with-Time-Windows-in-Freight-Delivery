#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import time

s = time.time()
# Importing googlemaps module
import googlemaps

# Requires API key
gmaps = googlemaps.Client(key='') #Insert key

dir_name = os.path.dirname(os.path.realpath('__file__'))
file ="9Aug19"

file_name = os.path.join(dir_name, 'Real Dataset\\',file +'.csv')
df = pd.read_csv(file_name, encoding ='latin1', error_bad_lines=False)

d = {'Name': 'first', 'Post Code': 'first', 'Date': 'first', 'Reference': 'first', 'Type': 'first', 'Value' : 'sum'}
df_new = df.groupby('No', as_index=False).aggregate(d).reindex(columns=df.columns)
print(df_new)

cust_size = df_new.shape[0]

postcode = df_new['Post Code']

dist_matrix = pd.DataFrame(0, index=range(cust_size), columns=range(cust_size))
time_matrix = pd.DataFrame(0, index=range(cust_size), columns=range(cust_size))

for i in range(cust_size):
    for j in range(cust_size):
        if i!= j:
            my_dist = gmaps.distance_matrix(postcode[i],postcode[j])['rows'][0]['elements'][0]
            #print(i, j, my_dist)
            dist_matrix[i][j] = my_dist['distance']['value']
            time_matrix[i][j] = my_dist['duration']['value']

e =time.time()

print(e-s)
dist_matrix.to_csv('Real Dataset\\9AugDist.csv', index=False, header=False)
time_matrix.to_csv('Real Dataset\\9AugTime.csv', index=False, header=False)



