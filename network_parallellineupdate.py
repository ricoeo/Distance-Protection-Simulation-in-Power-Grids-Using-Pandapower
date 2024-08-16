import pandapower as pp
import pandas as pd
import pandapower.topology as top
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
import pandapower.shortcircuit as sc

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# Select sheets to read
bus_data = pd.read_excel(excel_file, sheet_name='bus_data', index_col=0)
load_data = pd.read_excel(excel_file, sheet_name='load_data', index_col=0)
line_data = pd.read_excel(excel_file, sheet_name='line_data', index_col=0)
external_grid_data = pd.read_excel(excel_file, sheet_name='external_grid_data', index_col=0)
wind_gen_data = pd.read_excel(excel_file, sheet_name='wind_gen_data', index_col=0)

net = pp.create_empty_network()

# define some constant parameters
HV = 110e3  # High Voltage side in volts
S_base = 100e6  # Base power in watts (100 MW)
S_sc_HV = 5e9  # Short-circuit power at HV side in watts (5 GW)

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
    def __init__(self, device_id, bus_id, first_line_id, replaced_line_id, net, depth=3):
        self.device_id = device_id
        self.bus_id = bus_id
        self.associated_line_id = first_line_id
        self.replaced_line_id = replaced_line_id
        self.net = net
        self.associated_zone_impedance = self.find_associated_lines(first_line_id, depth)

    def update_associated_line(self):
        if not math.isnan(self.replaced_line_id):
            self.associated_line_id = self.replaced_line_id

    def find_associated_lines(self, start_line_id, depth):
        # Create a graph of the network
        graph = top.create_nxgraph(self.net, include_lines=True, include_impedances=True, calc_branch_impedances=True,
                                   multi=True, include_out_of_service=False)
        start_bus = self.bus_id
        # line_impedance=None
        if start_bus == self.net.line.at[start_line_id, 'from_bus']:
            next_bus = self.net.line.at[start_line_id, 'to_bus']
            # line_impedance = graph.get_edge_data(start_bus, next_bus)["r_ohm"]+1j **graph.get_edge_data(
            # start_line_id, next_bus)["x_ohm"]
        else:
            next_bus = self.net.line.at[start_line_id, 'from_bus']
            # line_impedance = graph.get_edge_data(start_bus, next_bus)["r_ohm"]+1j **graph.get_edge_data(start_line_id, next_bus)["x_ohm"]

        line_value = list(graph.get_edge_data(start_bus, next_bus).values())[0]
        first_line_impedance = line_value["r_ohm"] + 1j * line_value["x_ohm"]
        first_zone_line_impedance = (line_value["r_ohm"] + 1j * line_value["x_ohm"]) * 0.9
        associated_lines = [first_zone_line_impedance]

        # Traverse from the second bus and find subsequent lines
        previous_bus = start_bus
        current_bus = next_bus
        second_step_reach_parallel_flag = False
        second_zone_line_impedance = 0
        second_line_impedance = 0
        third_zone_line_impedance = 0
        current_depth_index = 1
        parallel_line_pair = [0,1]
        parallel_line_flag = True
        while current_depth_index < depth:
            min_weight = float('inf')
            current_depth_index += 1
            # max_line_id = None
            # Get all connected lines to the current bus
            edges = graph.edges(current_bus, data=True)

            if current_depth_index == 2:
                for from_bus, to_bus, data in edges:
                    if to_bus == previous_bus:
                        continue

                    # Check if there are parallel lines between from_bus and to_bus
                    parallel_line_flag = len([e for e in edges if (e[0] == from_bus and e[1] == to_bus) or (e[0] == to_bus and e[1] == from_bus)]) > 1

                    if parallel_line_flag:
                        if data['weight']*0.5 < min_weight:
                            second_step_reach_parallel_flag = True
                            min_weight = data['weight']*0.5
                            second_line_impedance = data["r_ohm"] + 1j * data["x_ohm"]
                            second_zone_line_impedance = 0.9 * (first_line_impedance+second_line_impedance * 0.5)
                            third_zone_line_impedance = 1.1 * (first_line_impedance+second_line_impedance)
                            next_bus = to_bus
                    else:
                        if data['weight'] < min_weight:
                            min_weight = data['weight']
                            second_line_impedance = data["r_ohm"] + 1j * data["x_ohm"]
                            second_zone_line_impedance = 0.9 * (first_line_impedance+second_line_impedance * 0.9)
                            next_bus = to_bus

            elif current_depth_index == 3:
                if second_zone_line_impedance is not None and second_step_reach_parallel_flag is not True:
                    for from_bus, to_bus, data in edges:
                        if to_bus == previous_bus:
                            continue

                        # Check for parallel line again at this depth
                        parallel_line_flag = len([e for e in edges if (e[0] == from_bus and e[1] == to_bus) or (
                                    e[0] == to_bus and e[1] == from_bus)]) > 1

                        if parallel_line_flag:
                            if data['weight']*0.5 < min_weight:
                                min_weight = data['weight']*0.5
                                third_zone_line_impedance = 0.9 * (first_line_impedance+second_line_impedance * 0.9 + (data["r_ohm"] + 1j * data["x_ohm"]) * 0.9 * 0.9 * 0.5)
                                next_bus = to_bus
                        else:
                            if data['weight'] < min_weight:
                                min_weight = data['weight']
                                third_zone_line_impedance = 0.9 * (first_line_impedance+second_line_impedance * 0.9 + (data["r_ohm"] + 1j * data["x_ohm"]) * 0.9 * 0.9 * 0.5)
                                next_bus = to_bus
            previous_bus = current_bus
            current_bus = next_bus

        associated_lines.append(second_zone_line_impedance)
        associated_lines.append(third_zone_line_impedance)

        return associated_lines

    def check_zone(self, impedance):
        """ Determine the protection zone based on impedance """
        # how to compare two complex numbers depends on our need
        r_arc = 2.5  # the arc compensation (ohm) value for 110kv for R-setting
        impedance_point = Point(impedance.real, impedance.imag)
        zone1_polygon = Polygon(
            [(0, 0), (
                -self.associated_zone_impedance[0].imag * math.tan(math.radians(30)),
                self.associated_zone_impedance[0].imag),
             (self.associated_zone_impedance[0].real + r_arc, self.associated_zone_impedance[0].imag), (
                 self.associated_zone_impedance[0].real + r_arc,
                 -(self.associated_zone_impedance[0].real + r_arc) * math.tan(math.radians(22)))])
        zone2_polygon = Polygon(
            [(0, 0), (
                -self.associated_zone_impedance[1].imag * math.tan(math.radians(30)),
                self.associated_zone_impedance[1].imag),
             (self.associated_zone_impedance[1].real + r_arc, self.associated_zone_impedance[1].imag), (
                 self.associated_zone_impedance[1].real + r_arc,
                 -(self.associated_zone_impedance[1].real + r_arc) * math.tan(math.radians(22)))])
        zone3_polygon = Polygon(
            [(0, 0), (
                -self.associated_zone_impedance[2].imag * math.tan(math.radians(30)),
                self.associated_zone_impedance[2].imag),
             (self.associated_zone_impedance[2].real + r_arc, self.associated_zone_impedance[2].imag), (
                 self.associated_zone_impedance[2].real + r_arc,
                 -(self.associated_zone_impedance[2].real + r_arc) * math.tan(math.radians(22)))])

        if zone1_polygon.contains(impedance_point) or zone1_polygon.touches(impedance_point):
            return "Zone 1"
        elif zone2_polygon.contains(impedance_point) or zone2_polygon.touches(impedance_point):
            return "Zone 2"
        elif zone3_polygon.contains(impedance_point) or zone3_polygon.touches(impedance_point):
            return "Zone 3"
        return "Out of Zone"

    def check_zone_with_mag_angle(self, magnitude, angle):
        """ Determine the protection zone based on impedance """
        # how to compare two complex numbers depends on our need

        impedance_point = Point(magnitude * math.cos(angle), magnitude * math.sin(angle))
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


#for calculating the zones, it is necessary to not consider the intermediate nodes
for line in net.line.itertuples():
    if net.bus.loc[line.from_bus, 'type'] == '"n"' or net.bus.loc[line.to_bus, 'type'] == '"n"':
        print(line.Index)
        net.line.at[line.Index, 'in_service'] = False

line_between_C_D = pp.create_line_from_parameters(net, from_bus=2, to_bus=3,
                                                  length_km=line_data.at[5, "length_km"] + line_data.at[
                                                      6, "length_km"] + line_data.at[
                                                                7, "length_km"] + line_data.at[8, "length_km"] +
                                                            line_data.at[9, "length_km"] +
                                                            line_data.at[10, "length_km"],
                                                  r_ohm_per_km=line_data.at[5, "r_ohm_per_km"],
                                                  x_ohm_per_km=line_data.at[5, "x_ohm_per_km"],
                                                  c_nf_per_km=line_data.at[5, "c_nf_per_km"],
                                                  r0_ohm_per_km=line_data.at[5, "r0_ohm_per_km"],
                                                  x0_ohm_per_km=line_data.at[5, "x0_ohm_per_km"],
                                                  c0_nf_per_km=line_data.at[5, "c0_nf_per_km"],
                                                  max_i_ka=line_data.at[5, "max_i_ka"],
                                                  parallel=line_data.at[5, "parallel"])

#distance_protection_data = pd.read_excel(excel_file, sheet_name='dist_protect_data _simple', index_col=0)
distance_protection_data = pd.read_excel(excel_file, sheet_name='dist_protect_data_complex', index_col=0)
Protection_devices = {}
for idx in distance_protection_data.index:
    # print(idx)
    Protection_devices[idx] = ProtectionDevice(device_id=distance_protection_data.at[idx, "device_id"],
                                               bus_id=distance_protection_data.at[idx, "bus_id"],
                                               first_line_id=distance_protection_data.at[idx, "first_line_id"],
                                               replaced_line_id=distance_protection_data.at[idx, "replaced_line_id"],
                                               net=net)

# print(
#     f"Zone Thresholds: Zone 1: {Protection_devices[1].associated_zone_impedance[0]}, Zone 2: {Protection_devices[1].associated_zone_impedance[1]}, Zone 3: {Protection_devices[1].associated_zone_impedance[2]}")


#print(Protection_devices[0].check_zone(2 + 1j))

""" after initialing the protection device parameter, the line between c and d is not needed anymore"""
# recover the line
net.line.drop(line_between_C_D, inplace=True)
for line in net.line.itertuples():
    if net.bus.loc[line.from_bus, 'type'] == '"n"' or net.bus.loc[line.to_bus, 'type'] == '"n"':
        print(line.Index)
        net.line.at[line.Index, 'in_service'] = True

# keep the associated_zone_impedance but replace the associated lines
for device in Protection_devices.values():
    device.update_associated_line()


def find_affected_devices(line_id, all_devices):
    affected_devices = all_devices
    return affected_devices


def filter_and_save_devices_by_line_id(devices, line_id):
    """Filters the list of ProtectionDevice objects based on the associated_line_id and saves the associated_line_id."""
    matching_devices = []
    saved_associated_line_ids = []

    for device in devices.values():
        if device.associated_line_id == line_id:
            matching_devices.append(device)
            saved_associated_line_ids.append(device.associated_line_id)

    return matching_devices, saved_associated_line_ids


def find_line_id_between_buses(bus_id, temp_bus, net):
    # Find the line_id that connects bus_id and temp_bus
    for idx, line in net.line.iterrows():
        if (line['from_bus'] == bus_id and line['to_bus'] == temp_bus) or \
                (line['from_bus'] == temp_bus and line['to_bus'] == bus_id):
            return idx
    return None  # Return None or raise an error if no such line is found


def temporally_update_associated_line_id(matching_devices, temp_bus, net):
    if matching_devices is not None:
        for device in matching_devices:
            new_line_id = find_line_id_between_buses(device.bus_id, temp_bus, net)
            # update the device's associated line id
            device.associated_line_id = new_line_id


def recover_associated_line_id(matching_devices, saved_associated_line_ids):
    if matching_devices is not None:
        idx = 0
        for device in matching_devices:
            # update the device's associated line id
            device.associated_line_id = saved_associated_line_ids[idx]
            idx += 1


def simulate_faults_along_line(net, line_id, affected_devices, interval_km=0.25):
    """Simulate faults along a line by adding temporary buses at specified intervals."""
    # Make original line out of service
    line = net.line.loc[line_id]
    line['in_service'] = False
    # evenly distribute the faults in the line
    line_length = line.length_km
    num_faults = int(line_length / interval_km) - 1
    # temporally change some parameters of the protection devices due to the fault for simple calculation
    #matching_devices = list(filter(lambda device: device.associated_line_id == line_id, affected_devices))
    matching_devices, saved_ids = filter_and_save_devices_by_line_id(affected_devices, line_id)
    # if the sensed results are different from the calculated results, it is recorded
    sense_wrong_dict = {}
    if num_faults > 0:

        fault_locations = [interval_km * i for i in range(1, num_faults)]

        for fault_location in fault_locations:
            # Create a temporary bus at fault location
            temp_bus = pp.create_bus(net, vn_kv=HV, type="n", name="fault_bus")
            # Split the line at the fault location,copy all the other parameters of the original line to the new line
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
            # after adding the fault bus into the network, simulate a three-phase short circuit at the temporary bus
            sc.calc_sc(net, fault="3ph", bus=temp_bus, branch_results=True, return_all_currents=True)
            #change the parameters of the protection device
            temporally_update_associated_line_id(matching_devices, temp_bus, net)
            for device in affected_devices:

                # according to the line length to calcualte the distance the protection devices supposed to sense
                impedance = calculate_impedance(net, affected_devices[device], affected_devices[device].bus_id,
                                                temp_bus)
                if impedance is None:
                    print(
                        f"Impedance calculation returned None for device {affected_devices[device].device_id} at bus {affected_devices[device].bus_id}. Skipping this fault scenario.")
                    continue  # Skip to the next fault scenario
                zone_calculated = affected_devices[device].check_zone(impedance)

                # Get the impedance at the protection device through the line results
                if affected_devices[device].bus_id == net.line.loc[affected_devices[device].associated_line_id][
                    "from_bus"]:
                    vm_pu = net.res_line_sc.loc[affected_devices[device].associated_line_id]["vm_from_pu"]
                    ikss_ka = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_from_ka"]
                    va_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["va_from_degree"]
                    ikss_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_from_degree"]
                elif affected_devices[device].bus_id == net.line.loc[affected_devices[device].associated_line_id][
                    "to_bus"]:
                    vm_pu = net.res_line_sc.loc[affected_devices[device].associated_line_id]["vm_to_pu"]
                    ikss_ka = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_to_ka"]
                    va_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["va_to_degree"]
                    ikss_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_to_degree"]
                else:
                    print("the line result is not existed")
                # calcualte the magnitude and the angle of the impedance
                r_sensed = vm_pu * HV * 1e-3 / ikss_ka
                angle_sensed = va_degree - ikss_degree
                zone_sensed = affected_devices[device].check_zone_with_mag_angle(r_sensed, angle_sensed)

                # compare if the zone_calculated the same as zone_sensed
                if zone_calculated != zone_sensed:
                    # Collect data in the dictionary if discrepancies exist
                    sense_wrong_dict[line_id] = {
                        "fault_location": {
                            "device_id": affected_devices[device].device_id,
                            "distance_from_which_bus": line.from_bus,
                            "distance_km": fault_location
                        },
                        "calculated_impedance": impedance,
                        "sensed_impedance": {
                            "magnitude": r_sensed,
                            "angle": angle_sensed
                        },
                        "zones": {
                            "calculated": zone_calculated,
                            "sensed": zone_sensed
                        }
                    }
                # print(
                #     f"Device {affected_devices[device].device_id} at Bus {affected_devices[device].bus_id} triggered: {zone_calculated}, Impedance: {impedance}")
            recover_associated_line_id(matching_devices, saved_ids)
            # Remove temporary buses and associated lines after analysis.
            net.line.drop(temp_line_part1, inplace=True)
            net.line.drop(temp_line_part2, inplace=True)
            net.bus.drop(temp_bus, inplace=True)

    return sense_wrong_dict


# def cleanup_temp_buses(net, temp_buses):
#     """Remove temporary buses and associated lines after analysis."""
#     for bus in temp_buses:
#         lines_to_remove = net.line[(net.line.from_bus == bus) | (net.line.to_bus == bus)].index
#         net.line.drop(lines_to_remove, inplace=True)
#         net.bus.drop(bus, inplace=True)

def convert_to_directed_old(g_undirected, initial_direction):
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


def convert_to_directed(g_undirected, initial_direction):
    g_directed = nx.DiGraph()  # Initialize a directed graph
    keys_to_extract = ['weight', 'r_ohm', 'x_ohm']

    # Copy all other edges from the undirected graph, preserving their attributes
    for u, v in g_undirected.edges():
        attrs_temp = g_undirected.get_edge_data(u, v)
        required_attrs_temp = {key: next(iter(attrs_temp.values()))[key] for key in keys_to_extract}
        g_directed.add_edge(u, v, **required_attrs_temp)
        g_directed.add_edge(v, u, **required_attrs_temp)

    for neighbor in g_undirected.neighbors(initial_direction[0]):
        if neighbor == initial_direction[1]:
            if g_directed.has_edge(initial_direction[1], initial_direction[0]):
                g_directed.remove_edge(initial_direction[1], initial_direction[0])
        else:
            if g_directed.has_edge(initial_direction[0], neighbor):
                g_directed.remove_edge(initial_direction[0], neighbor)

    return g_directed


def calculate_impedance(net, device, from_bus, to_bus):
    """Calculate the impedance between two buses. Shortest distance"""
    graph = top.create_nxgraph(net, include_lines=True, include_impedances=True, calc_branch_impedances=True)
    initial_associated_line = net.line.loc[device.associated_line_id]
    initial_from_bus = from_bus
    initial_to_bus = initial_associated_line.from_bus if initial_from_bus != initial_associated_line.from_bus else initial_associated_line.to_bus

    # Set initial direction based on device's associated line
    initial_direction = (initial_from_bus, initial_to_bus)
    directed_graph = convert_to_directed(graph, initial_direction)

    total_impedance = 0
    try:
        # Ensure the path exists and retrieve the shortest path based on the impedance
        path = nx.dijkstra_path(directed_graph, source=from_bus, target=to_bus, weight='weight')
        # start to consider several complicated cases related with parallel lines
        # case 1 bypassing the external grid
        # Get bus numbers connected to the external grid
        external_grid_buses = net.ext_grid['bus'].tolist()

        # Check if the path bypasses any external grid bus
        for bus in external_grid_buses:
            if bus in path and path.index(bus) != 0 and path.index(bus) != len(path) - 1:
                print(
                    f"Path bypasses one of the external grid buses. Device {device.device_id} should not be considered.")
                return None

        path_bus_pairs = set(tuple(sorted([u, v])) for u, v in zip(path[:-1], path[1:]))
        # Identify all bus pairs with parallel lines
        parallel_lines = net.line[net.line.duplicated(subset=['from_bus', 'to_bus'], keep=False)]
        parallel_bus_pairs = set(
            tuple(sorted([row['from_bus'], row['to_bus']])) for _, row in parallel_lines.iterrows())
        # right now the 0 and 1 is hard coded, it will be fixed later
        path_involves_segments_other_than_0_to_bus = len(path_bus_pairs.difference({tuple(sorted([0, to_bus]))})) > 0
        path_involves_segments_other_than_1_to_bus = len(path_bus_pairs.difference({tuple(sorted([1, to_bus]))})) > 0
        parallel_in_path = path_bus_pairs.intersection(parallel_bus_pairs)
        for u, v in zip(path[:-1], path[1:]):
            bus_pair = tuple(sorted([u, v]))
            # Case 2 if parallel line is in path
            if parallel_in_path is not None and bus_pair in parallel_in_path:
                parallel_lines_for_pair = parallel_lines[
                    (parallel_lines['from_bus'] == bus_pair[0]) &
                    (parallel_lines['to_bus'] == bus_pair[1])
                    ]

                # Calculate combined impedance of parallel lines
                combined_impedance_r = directed_graph.get_edge_data(u, v)["r_ohm"] * 0.5
                combined_impedance_x = directed_graph.get_edge_data(u, v)["x_ohm"] * 0.5
                total_impedance += combined_impedance_r + 1j * combined_impedance_x
            # Case 3 Fault on Parallel Line
            elif tuple(sorted([0, to_bus])) == bus_pair and path_involves_segments_other_than_0_to_bus:
                line_part1_value = list(graph.get_edge_data(0, to_bus).values())[0]
                line_part2_value = list(graph.get_edge_data(1, to_bus).values())[0]
                line_part3_value = list(graph.get_edge_data(0, 1).values())[0]
                combined_impedance_r = 1/(1 / (line_part1_value["r_ohm"] + 1e-12) + 1 / (
                        line_part2_value["r_ohm"] + line_part3_value["r_ohm"] + 1e-12))
                combined_impedance_x = 1/(1 / (line_part1_value["x_ohm"] + 1e-12) + 1 / (
                        line_part2_value["x_ohm"] + line_part3_value["x_ohm"] + 1e-12))
                total_impedance += combined_impedance_r + 1j * combined_impedance_x
            elif tuple(sorted([1, to_bus])) == bus_pair and path_involves_segments_other_than_1_to_bus:
                line_part1_value = list(graph.get_edge_data(1, to_bus).values())[0]
                line_part2_value = list(graph.get_edge_data(0, to_bus).values())[0]
                line_part3_value = list(graph.get_edge_data(0, 1).values())[0]
                combined_impedance_r = 1/(1 / (line_part1_value["r_ohm"] + 1e-12) + 1 / (
                        line_part2_value["r_ohm"] + line_part3_value["r_ohm"] + 1e-12))
                combined_impedance_x = 1/(1 / (line_part1_value["x_ohm"] + 1e-12) + 1 / (
                        line_part2_value["x_ohm"] + line_part3_value["x_ohm"] + 1e-12))
                total_impedance += combined_impedance_r + 1j * combined_impedance_x
            else:
                line_value = directed_graph.get_edge_data(u, v)
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
        fault_simulation_results = simulate_faults_along_line(net, line_id, affected_devices, interval_km=0.25)

        # Restore the original line to service after analysis
        net.line.at[line_id, 'in_service'] = True

print(fault_simulation_results[0])
