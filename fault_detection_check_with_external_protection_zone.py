from shared.library import *
from shared.protection_border_adjust import adjust_protection_zone_with_external_parameters as adjust_protection_zone_ep

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# a new create network function is written
net_without_BC = create_network_without_BC(excel_file)

# change the protection zone setting according to the external parameters
Protection_devices = setup_protection_zones(net_without_BC, excel_file)
external_parameter_file = 'protection_zones_external_parameters_1211.xlsx'
Protection_devices_fix = adjust_protection_zone_ep(external_parameter_file,Protection_devices)

# simulate faults
protection_data= simulate_faults_for_all_lines(net_without_BC, Protection_devices_fix)

# Convert the list of dictionaries into a DataFrame
protection_df = pd.DataFrame(protection_data)


# write to Excel file
protection_df.to_excel('fault_detection_with_new_protection_border_1311.xlsx', index=False)
