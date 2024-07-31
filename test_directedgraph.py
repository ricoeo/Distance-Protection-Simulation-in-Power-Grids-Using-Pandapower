import networkx as nx

def setup_directed_graph_with_initial_direction(edges, initial_direction):
    G = nx.DiGraph()  # Create a directed graph
    # Set the initial direction
    G.add_edge(*initial_direction)

    # Add remaining edges without direction first
    for edge in edges:
        if tuple(edge) != tuple(initial_direction) and tuple(edge[::-1]) != tuple(initial_direction):
            G.add_edge(*edge, bidirectional=True)

    # Propagate directionality
    to_visit = [initial_direction[1]]  # Start from the target of the initial direction
    while to_visit:
        current = to_visit.pop(0)
        neighbors = [n for n in G[current] if G[current][n].get('bidirectional')]
        for neighbor in neighbors:
            # Set direction based on the current node to the neighbor
            G[current][neighbor].pop('bidirectional', None)  # Remove the bidirectional flag
            G.add_edge(current, neighbor)
            to_visit.append(neighbor)

    return G

# Define all edges (including the new intermediate bus 5 between bus 4 and bus 1)
edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (2, 4), (3, 1)]

# Define the initial direction
initial_direction = (1, 2)

# Setup the graph
G = setup_directed_graph_with_initial_direction(edges, initial_direction)

# Print the directed edges to confirm the setup
print("Directed Edges:", list(G.edges()))

# Calculate the shortest path from the new intermediate bus (bus 5) to bus 1
try:
    shortest_path = nx.shortest_path(G, source=1, target=5)
    print("Shortest path from bus 5 to bus 1:", shortest_path)
except nx.NetworkXNoPath:
    print("No path found from bus 5 to bus 1.")
