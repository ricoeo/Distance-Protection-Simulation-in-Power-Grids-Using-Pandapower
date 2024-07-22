import pandapower as pp
import pandas as pd
from pandapower.plotting import simple_plotly
from pandapower.plotting import simple_plot
import pandapower.topology as top
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# Select sheets to read
bus_data = pd.read_excel(excel_file, sheet_name='bus_data', index_col=0)
load_data = pd.read_excel(excel_file, sheet_name='load_data', index_col=0)
line_data = pd.read_excel(excel_file, sheet_name='line_data', index_col=0)
external_grid_data = pd.read_excel(excel_file, sheet_name='external_grid_data', index_col=0)
wind_gen_data = pd.read_excel(excel_file, sheet_name='wind_gen_data', index_col=0)

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

print(net)


class ProtectionDevice:
    def __init__(self, device_id, bus_id, first_line_id, net, depth=3):
        self.device_id = device_id
        self.bus_id = bus_id
        self.net = net
        self.associated_zone_impedance = self.find_associated_lines(first_line_id, depth)

    def find_associated_lines(self, start_line_id, depth):
        # Create a graph of the network
        graph = top.create_nxgraph(self.net, include_lines=True, include_impedances=True, calc_branch_impedances=True,
                                   multi=True, include_out_of_service=False)
        start_bus = self.bus_id
        #line_impedance=None
        if start_bus == self.net.line.at[start_line_id, 'from_bus']:
            next_bus = self.net.line.at[start_line_id, 'to_bus']
            # line_impedance = graph.get_edge_data(start_bus, next_bus)["r_ohm"]+1j **graph.get_edge_data(
            # start_line_id, next_bus)["x_ohm"]
        else:
            next_bus = self.net.line.at[start_line_id, 'from_bus']
            #line_impedance = graph.get_edge_data(start_bus, next_bus)["r_ohm"]+1j **graph.get_edge_data(start_line_id, next_bus)["x_ohm"]

        line_value = list(graph.get_edge_data(start_bus, next_bus).values())[0]
        first_zone_line_impedance = (line_value["r_ohm"] + 1j * line_value["x_ohm"]) * 0.9
        associated_lines = [first_zone_line_impedance]

        # Traverse from the second bus and find subsequent lines
        previous_bus = start_bus
        current_bus = next_bus
        second_zone_line_impedance = 0
        third_zone_line_impedance = 0
        current_depth_index = 1
        while current_depth_index < depth:
            max_weight = 0
            current_depth_index += 1
            #max_line_id = None
            # Get all connected lines to the current bus
            edges = graph.edges(current_bus, data=True)

            for from_bus, to_bus, data in edges:
                #line_resistance_0 = data["r_ohm"]
                if to_bus != previous_bus:
                    #choose the line which has highes weight and add it into the associated lines
                    if data['weight'] > max_weight:
                        max_weight = data['weight']
                        next_bus = to_bus
                        if current_depth_index == 2:
                            second_zone_line_impedance = 0.9 * (
                                    first_zone_line_impedance + data["r_ohm"] + 1j * data["x_ohm"])
                        elif current_depth_index == 3:
                            if second_zone_line_impedance is not None:
                                third_zone_line_impedance = 0.9 * (
                                        second_zone_line_impedance + data["r_ohm"] + 1j * data["x_ohm"])
            previous_bus = current_bus
            current_bus = next_bus

        associated_lines.append(second_zone_line_impedance)
        associated_lines.append(third_zone_line_impedance)

        return associated_lines

    def check_zone(self, impedance):
        """ Determine the protection zone based on impedance """
        #how to compare two complex numbers depends on our need

        impedance_point = Point(impedance.real, impedance.imag)
        zone1_polygon = Polygon(
            [(0, 0), (
                -self.associated_zone_impedance[0].imag * math.tan(math.radians(30)),
                self.associated_zone_impedance[0].imag),
             (self.associated_zone_impedance[0].real, self.associated_zone_impedance[0].imag), (
                 self.associated_zone_impedance[0].real,
                 -self.associated_zone_impedance[0].real * math.tan(math.radians(22)))])
        zone2_polygon = Polygon(
            [(0, 0), (
                -self.associated_zone_impedance[1].imag * math.tan(math.radians(30)),
                self.associated_zone_impedance[1].imag),
             (self.associated_zone_impedance[1].real, self.associated_zone_impedance[1].imag), (
                 self.associated_zone_impedance[1].real,
                 -self.associated_zone_impedance[1].real * math.tan(math.radians(22)))])
        zone3_polygon = Polygon(
            [(0, 0), (
                -self.associated_zone_impedance[2].imag * math.tan(math.radians(30)),
                self.associated_zone_impedance[2].imag),
             (self.associated_zone_impedance[2].real, self.associated_zone_impedance[2].imag), (
                 self.associated_zone_impedance[2].real,
                 -self.associated_zone_impedance[2].real * math.tan(math.radians(22)))])

        if zone1_polygon.contains(impedance_point) or zone1_polygon.touches(impedance_point):
            return "Zone 1"
        elif zone2_polygon.contains(impedance_point) or zone2_polygon.touches(impedance_point):
            return "Zone 2"
        elif zone3_polygon.contains(impedance_point) or zone3_polygon.touches(impedance_point):
            return "Zone 3"
        return "Out of Zone"


# #for calculating the zones, it is neccessary to not consider the intermediate nodes
# for line in net.line.itertuples():
#     if net.bus.loc[line.from_bus, 'type'] == '"n"' or net.bus.loc[line.to_bus, 'type'] == '"n"':
#         print(line.Index)
#         net.line.at[line.Index, 'in_service'] = False
#
# pp.create_line_from_parameters(net, from_bus=2, to_bus=3,
#                                length_km=line_data.at[5, "length_km"] + line_data.at[6, "length_km"] + line_data.at[
#                                    7, "length_km"] + line_data.at[8, "length_km"] + line_data.at[9, "length_km"] +
#                                          line_data.at[10, "length_km"], r_ohm_per_km=line_data.at[5, "r_ohm_per_km"],
#                                x_ohm_per_km=line_data.at[5, "x_ohm_per_km"], c_nf_per_km=line_data.at[5, "c_nf_per_km"],
#                                r0_ohm_per_km=line_data.at[5, "r0_ohm_per_km"],
#                                x0_ohm_per_km=line_data.at[5, "x0_ohm_per_km"],
#                                c0_nf_per_km=line_data.at[5, "c0_nf_per_km"], max_i_ka=line_data.at[5, "max_i_ka"],
#                                parallel=line_data.at[5, "parallel"])

distance_protection_data = pd.read_excel(excel_file, sheet_name='dist_protect_data _simple', index_col=0)
Protection_devices = {}
for idx in distance_protection_data.index:
    # print(idx)
    Protection_devices[idx] = ProtectionDevice(device_id=distance_protection_data.at[idx, "device_id"],
                                               bus_id=distance_protection_data.at[idx, "bus_id"],
                                               first_line_id=distance_protection_data.at[idx, "first_line_id"], net=net)

print(
    f"Zone Thresholds: Zone 1: {Protection_devices[1].associated_zone_impedance[0]}, Zone 2: {Protection_devices[1].associated_zone_impedance[1]}, Zone 3: {Protection_devices[1].associated_zone_impedance[2]}")
print(Protection_devices[0].check_zone(2 + 1j))


def find_affected_devices(line_id, all_devices):
    affected_devices = all_devices
    return affected_devices


def simulate_faults_along_line(net, line_id, affected_devices, interval_km=0.25):
    """Simulate faults along a line by adding temporary buses at specified intervals."""
    net.line.at[line_id, 'in_service'] = False
    line = net.line.loc[line_id]
    line_length = line.length_km
    num_faults = int(line_length / interval_km) - 1
    if num_faults > 0:
        fault_locations = [interval_km * i for i in range(1, num_faults)]

        for fault_location in fault_locations:
            # Create a temporary bus at fault location
            temp_bus = pp.create_bus(net, vn_kv=110, type="n", name="fault_bus")
            # Split the line at the fault location,copy all the other parameters of the original line to the new line, except for the length
            temp_line_part1 = pp.create_line_from_parameters(net, from_bus=line.from_bus, to_bus=temp_bus,
                                                             length_km=fault_location,
                                                             **{attr: getattr(line, attr) for attr in line_data.columns
                                                                if attr not in ['from_bus', 'to_bus', 'length_km',
                                                                                'name']})
            temp_line_part2 = pp.create_line_from_parameters(net, from_bus=temp_bus, to_bus=line.to_bus,
                                                             length_km=line_length - fault_location,
                                                             **{attr: getattr(line, attr) for attr in line_data.columns
                                                                if attr not in ['from_bus', 'to_bus', 'length_km',
                                                                                'name']})
            for device in affected_devices:
                impedance = calculate_impedance(net, temp_bus, affected_devices[device].bus_id)
                zone_calculated = affected_devices[device].check_zone(impedance)
                print(
                    f"Device {affected_devices[device].device_id} at Bus {affected_devices[device].bus_id} triggered: {zone_calculated}, Impedance: {impedance}")

            net.line.drop(temp_line_part1, inplace=True)
            net.line.drop(temp_line_part2, inplace=True)
            net.bus.drop(temp_bus, inplace=True)
            # Make original line out of service


# def cleanup_temp_buses(net, temp_buses):
#     """Remove temporary buses and associated lines after analysis."""
#     for bus in temp_buses:
#         lines_to_remove = net.line[(net.line.from_bus == bus) | (net.line.to_bus == bus)].index
#         net.line.drop(lines_to_remove, inplace=True)
#         net.bus.drop(bus, inplace=True)


def calculate_impedance(net, from_bus, to_bus):
    """Calculate the impedance between two buses. Shortest distance"""
    graph = top.create_nxgraph(net, include_lines=True, include_impedances=True, calc_branch_impedances=True)
    total_impedance = 0
    try:
        # Ensure the path exists and retrieve the shortest path based on the impedance
        path = nx.dijkstra_path(graph, source=from_bus, target=to_bus, weight='weight')
        # Calculate total impedance by summing the complex impedance of each line in the path
        for u, v in zip(path[:-1], path[1:]):
            line_value = list(graph.get_edge_data(u, v).values())[0]
            total_impedance += line_value["r_ohm"] + 1j * line_value["x_ohm"]
    except nx.NetworkXNoPath:
        print(f"No path available from bus {from_bus} to bus {to_bus}.")
        total_impedance = None  # Indicate that no path exists.
    return total_impedance


for line_id in net.line.index:
    if net.line.at[line_id, 'in_service']:  # Only consider in-service lines
        print(f"Simulating faults along line {line_id}")

        # Assuming that affected devices are consistent along the line
        affected_devices = find_affected_devices(line_id, Protection_devices)

        # Simulate faults along the line
        simulate_faults_along_line(net, line_id, affected_devices, interval_km=0.25)

        # Restore the original line to service after analysis
        net.line.at[line_id, 'in_service'] = True

