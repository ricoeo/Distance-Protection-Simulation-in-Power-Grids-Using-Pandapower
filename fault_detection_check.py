from shared.library import *

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

# net_with_gen = create_network(excel_file)
# net_without_gen = create_network_without_gen(excel_file)
# net_withou_BC = create_network_without_BC(excel_file)
# net_without_AB1 = create_network_without_AB1(excel_file)
# net_meshed_simple =create_network_meshed_simple(excel_file)
net = create_network(excel_file)

# set the protection zone setting
Protection_devices = setup_protection_zones(net, excel_file)

# Call the function to simulate faults and get protection data
protection_data = simulate_faults_for_all_lines(net, Protection_devices)

# Convert the list of dictionaries into a DataFrame
protection_df = pd.DataFrame(protection_data)

# write to Excel file
protection_df.to_excel("fault_detection_check_zero_generation.xlsx", index=False)
