from shared.library import *


def simulate_faults_for_half_line_position(
    net, protection_devices, plot_power_flag=False
):
    """Simulate faults along all in-service lines and collect data for protection devices."""
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
            affected_devices = find_affected_devices(line_id, protection_devices, net)
            # Set the flag if the current line_id is part of any double-line pair
            fault_line_on_doubleline_flag = line_id in double_line_ids

            # Simulate faults along the line
            simulate_faults_along_line(
                net,
                line_id,
                affected_devices,
                fault_line_on_doubleline_flag,
                half_line_position,
                plot_power_flag,
            )

            # Restore the original line to service after analysis
            net.line.at[line_id, "in_service"] = True


# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

# a new create network function is written
net = create_network_without_BE(excel_file)

Protection_devices = setup_protection_zones(net, excel_file)

# uncomment the savefig in the plot_short_circuit_results function to save figures into folders
simulate_faults_for_half_line_position(net, Protection_devices, plot_power_flag=True)

plt.show()
