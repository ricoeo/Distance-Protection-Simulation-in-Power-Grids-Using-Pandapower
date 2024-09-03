import pandapower as pp
import pandas as pd
import pandapower.topology as top
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
        first_zone_line_impedance = first_line_impedance * 0.9
        associated_lines = [first_zone_line_impedance]

        # Variables to track impedance at different depths
        second_line_impedance = 0
        second_zone_line_impedance = None
        third_zone_line_impedance = None
        second_depth_buses = []
        current_depth_index = 1

        # Traverse from the second bus and find subsequent lines
        previous_bus = start_bus
        current_bus = next_bus

        second_step_reach_parallel_flag = False

        parallel_line_pair = [0, 1]
        parallel_line_flag = True
        while current_depth_index < depth:
            min_weight = float('inf')
            current_depth_index += 1

            if current_depth_index == 2:
                depth_2_edges = graph.edges(current_bus, data=True)

                for from_bus, to_bus, data in depth_2_edges:
                    if to_bus == previous_bus:
                        continue  # Avoid going backward

                    # Store all buses and impedances for depth 2
                    second_depth_buses.append(to_bus)

                    # Check if there are parallel lines between from_bus and to_bus
                    parallel_line_flag = len([e for e in depth_2_edges if (e[0] == from_bus and e[1] == to_bus) or (
                                e[0] == to_bus and e[1] == from_bus)]) > 1

                    if parallel_line_flag:
                        if data['weight'] * 0.5 < min_weight:
                            second_step_reach_parallel_flag = True
                            min_weight = data['weight'] * 0.5
                            second_line_impedance = data["r_ohm"] + 1j * data["x_ohm"]
                            second_zone_line_impedance = 0.9 * (first_line_impedance + second_line_impedance * 0.5)
                            third_zone_line_impedance = 1.1 * (first_line_impedance + second_line_impedance)
                            # next_bus = to_bus
                    else:
                        if data['weight'] < min_weight:
                            min_weight = data['weight']
                            second_line_impedance = data["r_ohm"] + 1j * data["x_ohm"]
                            second_zone_line_impedance = 0.9 * (first_line_impedance + second_line_impedance * 0.9)
                            # next_bus = to_bus

            elif current_depth_index == 3:
                if second_zone_line_impedance is not None and second_step_reach_parallel_flag is not True:
                    for bus_index in second_depth_buses:
                        depth_3_edges = graph.edges(bus_index, data=True)
                        for from_bus, to_bus, data in depth_3_edges:
                            if to_bus == previous_bus:
                                continue

                            # Check for parallel line again at this depth
                            parallel_line_flag = len([e for e in depth_3_edges if (e[0] == from_bus and e[1] == to_bus) or (
                                    e[0] == to_bus and e[1] == from_bus)]) > 1

                            if parallel_line_flag:
                                if data['weight'] * 0.5 < min_weight:
                                    min_weight = data['weight'] * 0.5
                                    third_zone_line_impedance = 0.9 * (
                                                first_line_impedance + second_line_impedance * 0.9 + (
                                                    data["r_ohm"] + 1j * data["x_ohm"]) * 0.9 * 0.9 * 0.5)
                            else:
                                if data['weight'] < min_weight:
                                    min_weight = data['weight']
                                    third_zone_line_impedance = 0.9 * (
                                                first_line_impedance + second_line_impedance * 0.9 + (
                                                    data["r_ohm"] + 1j * data["x_ohm"]) * 0.9 * 0.9)
            # else:
            #     print("Please implement a new zone-grading algorithm.")
            previous_bus = current_bus

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
                                                  length_km=line_data.at[6, "length_km"] + line_data.at[
                                                      7, "length_km"] + line_data.at[
                                                                8, "length_km"] + line_data.at[9, "length_km"] +
                                                            line_data.at[10, "length_km"] +
                                                            line_data.at[11, "length_km"],
                                                  r_ohm_per_km=line_data.at[6, "r_ohm_per_km"],
                                                  x_ohm_per_km=line_data.at[6, "x_ohm_per_km"],
                                                  c_nf_per_km=line_data.at[6, "c_nf_per_km"],
                                                  r0_ohm_per_km=line_data.at[6, "r0_ohm_per_km"],
                                                  x0_ohm_per_km=line_data.at[6, "x0_ohm_per_km"],
                                                  c0_nf_per_km=line_data.at[6, "c0_nf_per_km"],
                                                  max_i_ka=line_data.at[6, "max_i_ka"],
                                                  parallel=line_data.at[6, "parallel"])

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

# Initialize a list to store the data
protection_data = []

# Loop through each ProtectionDevice and extract the associated zone impedances and line IDs
for idx, device in Protection_devices.items():
    protection_data.append({
        'Device ID': device.device_id,
        'Bus ID': device.bus_id,
        'First Line ID': device.associated_line_id,
        'Replaced Line ID': device.replaced_line_id,
        'Zone 1 Impedance': device.associated_zone_impedance[0],
        'Zone 2 Impedance': device.associated_zone_impedance[1],
        'Zone 3 Impedance': device.associated_zone_impedance[2]
    })

# # Convert the data into a DataFrame
protection_df = pd.DataFrame(protection_data)
#
# # Display the DataFrame to check the data
# print(protection_df)

# Save the DataFrame to an Excel file
output_file = 'protection_zones_revise_0903_wrong.xlsx'
protection_df.to_excel(output_file, index=False)

#print(f"Data saved to {output_file}")
