
import matplotlib.pyplot as plt
from shared.library import *

# Read the data from the excel file
excel_file = 'grid_data_sheet.xlsx'

net = create_network(excel_file)
Protection_devices = setup_protection_zones(net, excel_file)


""" Here it starts"""

line = net.line.loc[1]
line_id = 1
fault_line_on_doubleline_flag = line_id in [0, 1]
net.line.loc[line_id]['in_service'] = False
line_length = line.length_km
temp_bus = pp.create_bus(net, vn_kv=HV, type="n", name="fault_bus")
fault_location = 0.5
# Split the line at the fault location,copy all the other parameters of the original line to the new line
temp_line_part1 = pp.create_line_from_parameters(net, from_bus=line.from_bus, to_bus=temp_bus,
                                                             length_km=fault_location, in_service=True,
                                                             **{attr: line[attr] for attr in line.index if
                                                                attr not in ['from_bus', 'to_bus', 'length_km', 'name',
                                                                             'std_type', 'in_service']})
temp_line_part2 = pp.create_line_from_parameters(net, from_bus=temp_bus, to_bus=line.to_bus,
                                                 length_km=line_length - fault_location, in_service=True,
                                                 **{attr: line[attr] for attr in line.index if
                                                    attr not in ['from_bus', 'to_bus', 'length_km', 'name',
                                                                 'std_type', 'in_service']})
graph = top.create_nxgraph(net, include_lines=True, include_impedances=True, calc_branch_impedances=True)
initial_direction = (0, 1)

directed_graph = convert_to_directed(graph, initial_direction,temp_bus,fault_line_on_doubleline_flag)
pos = nx.shell_layout(directed_graph)

# Plot the graph with uniform edge lengths
plt.figure(figsize=(10, 7))
nx.draw_networkx(directed_graph, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='k',linewidths=2, font_size=15, arrows=True, arrowsize=20, width=2)

plt.show()
