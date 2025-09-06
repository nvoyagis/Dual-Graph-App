import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
import numpy as np
from datetime import datetime
# import Portfolio_Optimization as po

def build_dual_from_faces(faces):
    """
    Build the dual graph from faces from the TMFG MATLAB code.
    Each face is a node labeled by its sorted vertex tuple.
    Edges connect nodes that share exactly 2 vertices.
    """
    dual = nx.Graph()

    # Add each face as a node for the dual graph
    for face in faces:
        face = tuple(sorted(face))
        dual.add_node(face)

    # Connect dual vertices with exactly two stocks in common
    for f1, f2 in combinations(faces, 2):
        if len(set(f1) & set(f2)) == 2:
            dual.add_edge(tuple(sorted(f1)), tuple(sorted(f2)))

    return dual