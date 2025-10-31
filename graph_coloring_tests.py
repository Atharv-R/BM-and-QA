import dwave_networkx as dnx
import numpy as np
import networkx as nx

import gcol 

def draw_zephyr_hidden_visible(G, vh_nodearray):
    '''
    Draws G as a zephyr graph topology, with visible / hidden nodes as labeled in vh_nodearray
    G: Networkx Graph
    vh_nodearray: numpy array of 0 and 1s -- 1 for visible, 0 for hidden

    hidden nodes will be blue, and visibles red. 
    '''
    vh_dict = {} #dnx.draw needs dict 

    #assigning large negative or positive value for v vs h
    for i, entry in enumerate(vh_nodearray):
        vh_dict[i] = (entry-0.5)*1000

    #using linear biases, which dnx can color code, 
    # to do the actual coloring 
    dnx.draw_zephyr(G, linear_biases=vh_dict)
    return None

K = 5
G = dnx.zephyr_graph(K)
n = G.number_of_nodes()
print('K =', K, 'n =', n, 'n/K =', n/K, 
      'n/(32*K^2) =', n/(32*K**2+16*K))

dnx.draw_zephyr(G)
print(G.number_of_nodes())


coloring_dict = gcol.node_coloring(G)

# build a numpy array where index i holds coloring_dict[i] (or -1 if missing)
max_key = max(coloring_dict.keys())
arr_len = max(G.number_of_nodes(), max_key + 1)
color_array = np.full(arr_len, -1, dtype=int)
for k, v in coloring_dict.items():
    color_array[int(k)] = int(v)

# number of colors used
k = int(color_array.max()) + 1
print(f"Graph colored with {k} colors.")

# draw using the array (aligned with G.nodes())
node_colors = [int(color_array[int(node)]) for node in G.nodes()]
dnx.draw_zephyr(G, node_color=node_colors, cmap='viridis')