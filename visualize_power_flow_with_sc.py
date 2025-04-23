from shared.library import *

"""
This script is used to visualize the power flow in a grid network, fault bus at the middle point is the default setting.
"""


def simulate_faults_along_line_for_test(
    net,
    line_id,
    affected_devices,
    fault_line_on_doubleline_info,
    interval_km=0.25,
    plot_powerflow_flag=False,
):
    """this function is used to generate the tikz color code for the short circuit current, it is only for visulization purpose"""
    device_data_dict = []
    # Make original line out of service
    line = net.line.loc[line_id]
    net.line.at[line_id, "in_service"] = False
    # evenly distribute the faults in the line
    line_length = line.length_km
    num_faults = int(line_length / interval_km) - 1
    # temporally change some parameters of the protection devices due to the fault. for simple calculation
    matching_devices, saved_ids = filter_and_save_devices_by_line_id(
        affected_devices, line_id
    )
    # if the fault bus needs to be plotted, the geodata is neccessary unless it will raise index Error
    start_geo_x, start_geo_y = (
        net.bus_geodata.at[net.line.at[line_id, "from_bus"], "x"],
        net.bus_geodata.at[net.line.at[line_id, "from_bus"], "y"],
    )
    end_geo_x, end_geo_y = (
        net.bus_geodata.at[net.line.at[line_id, "to_bus"], "x"],
        net.bus_geodata.at[net.line.at[line_id, "to_bus"], "y"],
    )
    # if the sensed results are different from the calculated results, it is recorded
    # sense_wrong_dict = {}
    if num_faults > 0:

        fault_locations = [interval_km * i for i in range(1, num_faults)] or [
            line_length / 2
        ]

        for fault_location in fault_locations:
            # Create a temporary bus at fault location
            temp_bus = pp.create_bus(
                net,
                vn_kv=HV,
                type="n",
                name="fault_bus",
                geodata=(
                    start_geo_x
                    + (end_geo_x - start_geo_x) * float(fault_location) / line_length,
                    start_geo_y
                    + (end_geo_y - start_geo_y) * float(fault_location) / line_length,
                ),
            )
            # Split the line at the fault location,copy all the other parameters of the original line to the new line
            temp_line_part1 = pp.create_line_from_parameters(
                net,
                from_bus=line.from_bus,
                to_bus=temp_bus,
                length_km=fault_location,
                in_service=True,
                **{
                    attr: line[attr]
                    for attr in line.index
                    if attr
                    not in [
                        "from_bus",
                        "to_bus",
                        "length_km",
                        "name",
                        "std_type",
                        "in_service",
                    ]
                },
            )
            temp_line_part2 = pp.create_line_from_parameters(
                net,
                from_bus=temp_bus,
                to_bus=line.to_bus,
                length_km=line_length - fault_location,
                in_service=True,
                **{
                    attr: line[attr]
                    for attr in line.index
                    if attr
                    not in [
                        "from_bus",
                        "to_bus",
                        "length_km",
                        "name",
                        "std_type",
                        "in_service",
                    ]
                },
            )
            pp.runpp(net)
            # after adding the fault bus into the network, simulate a three-phase short circuit at the temporary bus
            sc.calc_sc(
                net,
                fault="3ph",
                bus=temp_bus,
                branch_results=True,
                return_all_currents=True,
                use_pre_fault_voltage=True,
            )
            if plot_powerflow_flag:
                generate_short_circuit_tikz_overlay(net, line_id, "original_net.txt")
            # change the parameters of the protection device
            temporally_update_associated_line_id(matching_devices, temp_bus, net)

            recover_associated_line_id(matching_devices, saved_ids)
            # Remove temporary buses and associated lines after analysis.
            net.line.drop(temp_line_part1, inplace=True)
            net.line.drop(temp_line_part2, inplace=True)
            net.bus.drop(temp_bus, inplace=True)

    return device_data_dict


def simulate_faults_for_half_line_position(
    net, protection_devices, plot_power_flag=False
):
    """Simulate faults at half position of all in-service lines and collect data for ploting short circuit current."""
    # Identify double lines by finding bus pairs with multiple lines in service
    double_line_pairs = (
        net.line[net.line["in_service"]]
        .groupby(["from_bus", "to_bus"])
        .filter(lambda x: len(x) > 1)
    )
    double_line_ids = set(double_line_pairs.index)
    for line_id in net.line.index:
        if net.line.at[line_id, "in_service"]:  # Only consider in-service lines
            half_line_position = net.line.at[line_id, "length_km"] / 2
            print(f"Simulating faults along line {line_id}")

            # Assuming that affected devices are consistent along the line
            affected_devices = protection_devices

            fault_line_on_doubleline_info = {}
            # Set the flag if the current line_id is part of any double-line pair
            fault_line_on_doubleline_info["flag"] = line_id in double_line_ids
            fault_line_on_doubleline_info["line_info"] = (
                net.line.at[line_id, "from_bus"],
                net.line.at[line_id, "to_bus"],
            )
            # Simulate faults along the line
            simulate_faults_along_line(
                net,
                line_id,
                affected_devices,
                fault_line_on_doubleline_info,
                half_line_position,
                plot_power_flag,
            )

            # Restore the original line to service after analysis
            net.line.at[line_id, "in_service"] = True


# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

# a new create network function is written
net = create_network(excel_file)

Protection_devices = setup_protection_zones(net, excel_file)

# uncomment the savefig in the plot_short_circuit_results function to save figures into folders
simulate_faults_for_half_line_position(net, Protection_devices, plot_power_flag=True)

plt.show()
