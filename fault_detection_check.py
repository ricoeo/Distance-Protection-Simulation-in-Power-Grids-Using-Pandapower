from shared.library import *

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# this is the short version, for every line four fault locations are analysed (if applicable)
# def simulate_faults_along_line(net, line_id, affected_devices, interval_km=0.25):
#     """Simulate faults along a line by adding temporary buses at specified intervals."""
#     device_data_dict = []
#     # Make original line out of service
#     line = net.line.loc[line_id]
#     line['in_service'] = False
#     # evenly distribute the faults in the line
#     line_length = line.length_km
#     num_faults = int(line_length / interval_km) - 1
#
#     # Temporally change some parameters of the protection devices due to the fault for simple calculation
#     matching_devices, saved_ids = filter_and_save_devices_by_line_id(affected_devices, line_id)
#
#     # Determine which fault locations to use
#     if num_faults > 2:
#         fault_locations = [interval_km * i for i in range(1, num_faults + 1)]
#         selected_fault_locations = fault_locations[:1] + fault_locations[-1:]
#     else:
#         selected_fault_locations = [interval_km * i for i in range(1, num_faults + 1)]
#
#     for fault_location in selected_fault_locations:
#         # Create a temporary bus at fault location
#         temp_bus = pp.create_bus(net, vn_kv=HV, type="n", name="fault_bus")
#         # Split the line at the fault location, copy all the other parameters of the original line to the new line
#         temp_line_part1 = pp.create_line_from_parameters(net, from_bus=line.from_bus, to_bus=temp_bus,
#                                                          length_km=fault_location,
#                                                          **{attr: getattr(line, attr) for attr in line_data.columns
#                                                             if attr not in ['from_bus', 'to_bus', 'length_km',
#                                                                             'name']})
#         temp_line_part2 = pp.create_line_from_parameters(net, from_bus=temp_bus, to_bus=line.to_bus,
#                                                          length_km=line_length - fault_location,
#                                                          **{attr: getattr(line, attr) for attr in line_data.columns
#                                                             if attr not in ['from_bus', 'to_bus', 'length_km',
#                                                                             'name']})
#         # after adding the fault bus into the network, simulate a three-phase short circuit at the temporary bus
#         sc.calc_sc(net, fault="3ph", bus=temp_bus, branch_results=True, return_all_currents=True)
#         # change the parameters of the protection device
#         temporally_update_associated_line_id(matching_devices, temp_bus, net)
#         for device in affected_devices:
#             # according to the line length to calculate the distance the protection devices supposed to sense
#             impedance = calculate_impedance(net, affected_devices[device], affected_devices[device].bus_id,
#                                             temp_bus, line_id)
#             if impedance is None:
#                 continue  # Skip to the next fault scenario
#             zone_calculated = affected_devices[device].check_zone(impedance)
#             # Get the impedance at the protection device through the line results
#             if affected_devices[device].bus_id == net.line.loc[affected_devices[device].associated_line_id][
#                 "from_bus"]:
#                 vm_pu = net.res_line_sc.loc[affected_devices[device].associated_line_id]["vm_from_pu"].item()
#                 ikss_ka = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_from_ka"].item()
#                 va_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["va_from_degree"].item()
#                 ikss_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id][
#                     "ikss_from_degree"].item()
#             elif affected_devices[device].bus_id == net.line.loc[affected_devices[device].associated_line_id][
#                 "to_bus"]:
#                 vm_pu = net.res_line_sc.loc[affected_devices[device].associated_line_id]["vm_to_pu"].item()
#                 ikss_ka = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_to_ka"].item()
#                 va_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["va_to_degree"].item()
#                 ikss_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_to_degree"].item()
#             else:
#                 print("The line result is not existed")
#             # calculate the magnitude and the angle of the impedance
#             r_sensed = vm_pu * HV * 1e-3 / ikss_ka
#             angle_sensed = va_degree - ikss_degree
#             zone_sensed = affected_devices[device].check_zone_with_mag_angle(r_sensed, angle_sensed)
#
#             # device_data = {
#             #     'Device ID': affected_devices[device].device_id,
#             #     'Fault_line_id': line_id,
#             #     'Referenced_bus': line.from_bus,
#             #     'Distance_from_bus': fault_location,
#             #     'Impedance_calculated': impedance,
#             #     'zone_calculated': zone_calculated,
#             #     'vm': vm_pu * HV,
#             #     'ikss': ikss_ka * 1e3,
#             #     'r_sensed': r_sensed,
#             #     'angle_sensed': angle_sensed,
#             #     'zone_sensed': zone_sensed,
#             #     'same_zone_detection': zone_calculated == zone_sensed
#             # }
#             device_data = {
#                 'Device ID': affected_devices[device].device_id,
#                 'Fault_line_id': line_id,
#                 'Referenced_bus': line.from_bus,
#                 'Distance_from_bus': fault_location,
#                 'zone_calculated': zone_calculated,
#                 'vm': vm_pu * HV,
#                 'ikss': ikss_ka * 1e3,
#                 'zone_sensed': zone_sensed,
#                 'same_zone_detection': zone_calculated == zone_sensed
#             }
#             # Append individual device data to the main list (flattened)
#             device_data_dict.append(device_data)
#         recover_associated_line_id(matching_devices, saved_ids)
#         # Remove temporary buses and associated lines after analysis.
#         net.line.drop(temp_line_part1, inplace=True)
#         net.line.drop(temp_line_part2, inplace=True)
#         net.bus.drop(temp_bus, inplace=True)
#
#     return device_data_dict

# another short version to anaylse fault happening position
# def simulate_faults_along_line(net, line_id, affected_devices, interval_km=0.25):
#     """Simulate faults along a line by adding temporary buses at specified intervals."""
#     device_data_dict = []
#     first_mismatch = None
#     last_mismatch = None
#     # fault on double line case is more complex
#     fault_line_on_doubleline_flag = line_id in [0, 1]
#     # Make original line out of service
#     line = net.line.loc[line_id]
#     line['in_service'] = False
#     # evenly distribute the faults in the line
#     line_length = line.length_km
#     num_faults = int(line_length / interval_km) - 1
#     # temporally change some parameters of the protection devices due to the fault for simple calculation
#     #matching_devices = list(filter(lambda device: device.associated_line_id == line_id, affected_devices))
#     matching_devices, saved_ids = filter_and_save_devices_by_line_id(affected_devices, line_id)
#     # if the sensed results are different from the calculated results, it is recorded
#     # sense_wrong_dict = {}
#     if num_faults > 0:
#
#         fault_locations = [interval_km * i for i in range(1, num_faults)]
#         for device in affected_devices:
#             for fault_location in fault_locations:
#                 # Create a temporary bus at fault location
#                 temp_bus = pp.create_bus(net, vn_kv=HV, type="n", name="fault_bus")
#                 # Split the line at the fault location,copy all the other parameters of the original line to the new line
#                 temp_line_part1 = pp.create_line_from_parameters(net, from_bus=line.from_bus, to_bus=temp_bus,
#                                                                  length_km=fault_location,
#                                                                  **{attr: getattr(line, attr) for attr in line_data.columns
#                                                                     if attr not in ['from_bus', 'to_bus', 'length_km',
#                                                                                     'name']})
#                 temp_line_part2 = pp.create_line_from_parameters(net, from_bus=temp_bus, to_bus=line.to_bus,
#                                                                  length_km=line_length - fault_location,
#                                                                  **{attr: getattr(line, attr) for attr in line_data.columns
#                                                                     if attr not in ['from_bus', 'to_bus', 'length_km',
#                                                                                     'name']})
#                 # after adding the fault bus into the network, simulate a three-phase short circuit at the temporary bus
#                 sc.calc_sc(net, fault="3ph", bus=temp_bus, branch_results=True, return_all_currents=True)
#                 #change the parameters of the protection device
#                 temporally_update_associated_line_id(matching_devices, temp_bus, net)
#
#                 # according to the line length to calcualte the distance the protection devices supposed to sense
#                 impedance = calculate_impedance(net, affected_devices[device], affected_devices[device].bus_id,
#                                                 temp_bus, fault_line_on_doubleline_flag)
#                 if impedance is None:
#                     # print(
#                     #     f"Impedance calculation returned None for device {affected_devices[device].device_id} at bus {affected_devices[device].bus_id}. Skipping this fault scenario.")
#                     continue  # Skip to the next fault scenario
#                 zone_calculated = affected_devices[device].check_zone(impedance)
#                 # Get the impedance at the protection device through the line results
#                 if affected_devices[device].bus_id == net.line.loc[affected_devices[device].associated_line_id][
#                     "from_bus"]:
#                     vm_pu = net.res_line_sc.loc[affected_devices[device].associated_line_id]["vm_from_pu"].item()
#                     ikss_ka = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_from_ka"].item()
#                     va_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id][
#                         "va_from_degree"].item()
#                     ikss_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id][
#                         "ikss_from_degree"].item()
#                 elif affected_devices[device].bus_id == net.line.loc[affected_devices[device].associated_line_id][
#                     "to_bus"]:
#                     vm_pu = net.res_line_sc.loc[affected_devices[device].associated_line_id]["vm_to_pu"].item()
#                     ikss_ka = net.res_line_sc.loc[affected_devices[device].associated_line_id]["ikss_to_ka"].item()
#                     va_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id]["va_to_degree"].item()
#                     ikss_degree = net.res_line_sc.loc[affected_devices[device].associated_line_id][
#                         "ikss_to_degree"].item()
#                 else:
#                     print("the line result is not existed")
#                 # calcualte the magnitude and the angle of the impedance
#                 r_sensed = vm_pu * HV * 1e-3 / ikss_ka
#                 angle_sensed = va_degree - ikss_degree
#                 zone_sensed = affected_devices[device].check_zone_with_mag_angle(r_sensed, angle_sensed)
#
#                 # Check if there is a mismatch in the zone detection
#                 if zone_calculated != zone_sensed:
#                     device_data = {
#                         'Device ID': affected_devices[device].device_id,
#                         'Fault_line_id': line_id,
#                         'Referenced_bus': line.from_bus,
#                         'Distance_from_bus': fault_location,
#                         'zone_calculated': zone_calculated,
#                         'vm': vm_pu * HV,
#                         'ikss': ikss_ka * 1e3,
#                         'zone_sensed': zone_sensed,
#                         'same_zone_detection': zone_calculated == zone_sensed
#                     }
#
#                     # Record the first mismatch
#                     if first_mismatch is None:
#                         first_mismatch = device_data
#                     # Always update the last mismatch
#                     last_mismatch = device_data
#
#                 recover_associated_line_id(matching_devices, saved_ids)
#                 # Remove temporary buses and associated lines after analysis.
#                 net.line.drop(temp_line_part1, inplace=True)
#                 net.line.drop(temp_line_part2, inplace=True)
#                 net.bus.drop(temp_bus, inplace=True)
#             # Add the first and last mismatch data (if they exist)
#             if first_mismatch:
#                 device_data_dict.append(first_mismatch)
#             if last_mismatch and last_mismatch != first_mismatch:
#                 device_data_dict.append(last_mismatch)
#
#     return device_data_dict


# output_file = 'fault_detection_check_without_gen.xlsx'
# Save the DataFrame to an Excel file

# Usage
net_with_gen = create_network(excel_file)
net_without_gen = create_network_without_gen(excel_file)

# the grid topology is not changed so the protection zone setting should be the same for both cases
Protection_devices = setup_protection_zones(net_with_gen, excel_file)

# Call the function to simulate faults and get protection data
protection_data_with_gen = simulate_faults_for_all_lines(net_with_gen, Protection_devices)
protection_data_without_gen = simulate_faults_for_all_lines(net_without_gen, Protection_devices)

# Convert the list of dictionaries into a DataFrame
protection_df_with_gen = pd.DataFrame(protection_data_with_gen)
protection_df_without_gen = pd.DataFrame(protection_data_without_gen)

# write to Excel file
protection_df_with_gen.to_excel('fault_detection_check_sc_sv.xlsx', index=False)
protection_df_without_gen.to_excel('protection_df_without_gen_sc_sv.xlsx', index=False)
