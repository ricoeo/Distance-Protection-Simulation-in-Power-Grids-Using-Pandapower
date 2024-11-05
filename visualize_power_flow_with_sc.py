from shared.library import *

def simulate_faults_for_half_line_position(net, protection_devices, plot_power_flag=False):
    """Simulate faults along all in-service lines and collect data for protection devices."""

    for line_id in net.line.index:
        if net.line.at[line_id, 'in_service']:  # Only consider in-service lines
            half_line_position = net.line.at[line_id, 'length_km']/2
            print(f"Simulating faults along line {line_id}")

            # Assuming that affected devices are consistent along the line
            affected_devices = find_affected_devices(line_id, protection_devices)

            # Simulate faults along the line
            simulate_faults_along_line(net, line_id, affected_devices, half_line_position, plot_power_flag)

            # Restore the original line to service after analysis
            net.line.at[line_id, 'in_service'] = True


# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# a new create network function is written
net = create_network_withoutparallelline(excel_file)

Protection_devices = setup_protection_zones(net, excel_file)

simulate_faults_for_half_line_position(net, Protection_devices, plot_power_flag=True)

plt.show()

