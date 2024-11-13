from shared.library import *

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

net_with_gen = create_network(excel_file)
net_without_gen = create_network_without_gen(excel_file)
net_withou_BC = create_network_without_BC(excel_file)

# the grid topology is not changed so the protection zone setting should be the same for both cases
Protection_devices = setup_protection_zones(net_withou_BC, excel_file)

# Call the function to simulate faults and get protection data
protection_data_without_BC = simulate_faults_for_all_lines(net_withou_BC, Protection_devices)

# Convert the list of dictionaries into a DataFrame
protection_df_without_BC = pd.DataFrame(protection_data_without_BC)

# write to Excel file
protection_df_without_BC.to_excel('fault_detection_check_1311_origin.xlsx', index=False)
