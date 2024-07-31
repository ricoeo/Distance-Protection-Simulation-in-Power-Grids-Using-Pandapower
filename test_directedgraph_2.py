import pandapower as pp
import pandas as pd
from pandapower.plotting import simple_plotly
from pandapower.plotting import simple_plot
import pandapower.topology as top
import networkx as nx

# Read the data from the excel file
excel_file = 'grid_data_sheet.xlsx'

# Select sheets to read
bus_data = pd.read_excel(excel_file, sheet_name='bus_data', index_col=0)
load_data = pd.read_excel(excel_file, sheet_name='load_data', index_col=0)
line_data = pd.read_excel(excel_file, sheet_name='line_data', index_col=0)
external_grid_data = pd.read_excel(excel_file, sheet_name='external_grid_data', index_col=0)
wind_gen_data = pd.read_excel(excel_file, sheet_name='wind_gen_data', index_col=0)

# create network as topology shows
net = pp.create_empty_network()
# buses
for idx in bus_data.index:
    pp.create_bus(net, vn_kv=bus_data.at[idx, "vn_kv"], name=bus_data.at[idx, "name"], type=bus_data.at[idx, "type"],
                  geodata=tuple(map(int, bus_data.at[idx, "geodata"].strip('()').split(','))))
# lines
for idx in line_data.index:
    pp.create_line_from_parameters(net, from_bus=line_data.at[idx, "from_bus"], to_bus=line_data.at[idx, "to_bus"],
                                   length_km=line_data.at[idx, "length_km"],
                                   r_ohm_per_km=line_data.at[idx, "r_ohm_per_km"],
                                   x_ohm_per_km=line_data.at[idx, "x_ohm_per_km"],
                                   c_nf_per_km=line_data.at[idx, "c_nf_per_km"],
                                   r0_ohm_per_km=line_data.at[idx, "r0_ohm_per_km"],
                                   x0_ohm_per_km=line_data.at[idx, "x0_ohm_per_km"],
                                   c0_nf_per_km=line_data.at[idx, "c0_nf_per_km"],
                                   max_i_ka=line_data.at[idx, "max_i_ka"], parallel=line_data.at[idx, "parallel"])
# loads
for idx in load_data.index:
    pp.create_load(net, bus=load_data.at[idx, "bus"], p_mw=load_data.at[idx, "p_mw"],
                   q_mvar=load_data.at[idx, "q_mvar"], name=load_data.at[idx, "name"])
# external grids
for idx in external_grid_data.index:
    pp.create_ext_grid(net, bus=external_grid_data.at[idx, "bus"], vm_pu=external_grid_data.at[idx, "vm_pu"],
                       va_degree=external_grid_data.at[idx, "va_degree"], name=external_grid_data.at[idx, "name"],
                       s_sc_max_mva=5e9, rx_max=0.1)

# generators
for idx in wind_gen_data.index:
    pp.create_sgen(net, bus=wind_gen_data.at[idx, "bus"], p_mw=wind_gen_data.at[idx, "p_mw"],
                   q_mvar=wind_gen_data.at[idx, "q_mvar"], sn_mva=wind_gen_data.at[idx, "sn_mva"],
                   name=wind_gen_data.at[idx, "name"], k=1.2)

def convert_to_directed(g_undirected, initial_direction):
    g_directed = nx.DiGraph()  # Initialize a directed graph
    # Add the initial directed edge with all attributes preserved
    attrs = g_undirected.get_edge_data(initial_direction[0], initial_direction[1])
    keys_to_extract = ['weight', 'r_ohm', 'x_ohm']
    if attrs:
        # Extract only the required attributes
        required_attrs = {key: next(iter(attrs.values()))[key] for key in keys_to_extract}
        g_directed.add_edge(initial_direction[0], initial_direction[1], **required_attrs)
        #checked_edge_data = g_directed.get_edge_data(initial_direction[0], initial_direction[1])

    # Copy all other edges from the undirected graph, preserving their attributes
    for u, v in g_undirected.edges():
        if (u, v) != initial_direction and (v, u) != initial_direction:
            attrs_temp = g_undirected.get_edge_data(u, v)
            required_attrs_temp = {key: next(iter(attrs_temp.values()))[key] for key in keys_to_extract}
            g_directed.add_edge(u, v, **required_attrs_temp)  # Add in original direction with attributes
            g_directed.add_edge(v, u, **required_attrs_temp)  # Add in reverse direction with attributes

    # Apply some rule or criteria to direct the graph from the initial edge
    # This is an example and can be adjusted based on specific rules or network behaviors
    visited = set([initial_direction[0]])
    stack = [initial_direction[1]]
    while stack:
        node = stack.pop(0)
        if node not in visited:
            visited.add(node)
            for successor in g_directed.successors(node):
                if successor not in visited:
                    stack.append(successor)
                    # Remove the reverse direction to enforce directedness
                    if g_directed.has_edge(successor, node):
                        g_directed.remove_edge(successor, node)

    return g_directed


graph = top.create_nxgraph(net, include_lines=True, include_impedances=True, calc_branch_impedances=True)
initial_direction = (0, 4)
directed_graph = convert_to_directed(graph, initial_direction)