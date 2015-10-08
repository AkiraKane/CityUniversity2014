# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:16:13 2014

@author: Daniel Dixey [ackf415]

INM430 CW01 Scripts
"""
################################################
# Any import statements you might need
import networkx as nx
import matplotlib.pyplot as plt
################################################
# DIY Exercise 2: Creating Graphs
################################################
G = nx.Graph()

G.add_edge('A', 'B', weight=0.1)
G.add_edge('B', 'C')
G.add_edge('C', 'A')
G.add_edge('E', 'C')
G.add_edge('D', 'C')
G.add_edge('F', 'D')
G.add_edge('H', 'F')
G.add_edge('G', 'F')
G.add_edge('I', 'J')

pos = nx.spring_layout(G)
print pos

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# Edges
nx.draw_networkx_edges(G, pos, width=6, edge_color='b')

# Labels
nx.draw_networkx_labels(G, pos, font_size=20)

plt.axis('off')
plt.show()

# Centrality
nx.degree_centrality(G)

# Betweenness
nx.edge_betweenness_centrality(G)

# Clustering Coefficient
nx.clustering(G)

# Simple Paths
for i in nx.all_simple_paths(G, 'A', 'F'):
    print i

# Shortest Path
nx.shortest_path(G, 'A', 'D')

# Number of Connected Components
nx.number_connected_components(G)

# Number of Nodes
for i in nx.connected_components(G):
    print i
################################################
