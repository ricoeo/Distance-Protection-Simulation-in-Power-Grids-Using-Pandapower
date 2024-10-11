from shared.library import *

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

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
