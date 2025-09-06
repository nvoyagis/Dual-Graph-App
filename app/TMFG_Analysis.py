import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
import scipy as sp
import cvxpy
import os
import time
from itertools import combinations
import random
import ta

# graph_dict keys: tuple of 4 dates (date1: beginning of data collection
#                                    date2: end of data collection
#                                    date3: beginning of selling period
#                                    date4: end of selling period)
# graph_dict values: tuple of graphs (graph1: TMFG
#                                     graph2: dual TMFG)
def find_frequent_edges(tmfg_list):
    # Count how many times each edge in the graphs of graph_list appear
    edge_counter_dict = {}
    for g in tmfg_list:
        for e in g.edges:
            if e in edge_counter_dict.keys():
                edge_counter_dict[e] += 1
            else:
                edge_counter_dict[e] = 1
    
    # Identify frequently-appearing edges
    important_edges = []
    for edge, counter in edge_counter_dict.items():
        if counter > 1:
            important_edges.append(edge)
    # Return important_edges
    print(important_edges)
    return important_edges
        
                        

'''
Simple case:
1) manually check how 2 TMFGs overlap
2) print similar edges
3) in simulatiosn function, track portfolios that contain both stocks that form the edge (track them by changing shape of nodes)
'''