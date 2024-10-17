from shared.library import *

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# a new create network function is written
net = create_network_withoutparallelline(excel_file)

# some changes have been done in the function find_associated_lines in the class and proetction device 4 ,5 is closed in the function find_affected_devices
Protection_devices = setup_protection_zones(net, excel_file)

# some changed have been done in the function calculate_impedance and convert_to_directed
protection_data= simulate_faults_for_all_lines(net, Protection_devices)

# Convert the list of dictionaries into a DataFrame
protection_df = pd.DataFrame(protection_data)

# write to Excel file
protection_df.to_excel('fault_detection_withoutparallel_line_prefault.xlsx', index=False)
