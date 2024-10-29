from shared.library import *
from shared.protection_border_adjust import adjust_protection_zone_with_measurement_data as adjust_protection_zone

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# a new create network function is written
net = create_network_withoutparallelline(excel_file)

# import the previous measurement data
excel_file_measure = 'fault_detection_withoutparallel_line.xlsx'
measurement_data = pd.read_excel(excel_file_measure)

# some changes have been done in the function find_associated_lines in the class and proetction device 4 ,5 is closed in the function find_affected_devices
Protection_devices = setup_protection_zones(net, excel_file)
Protection_devices_fix = adjust_protection_zone(measurement_data,Protection_devices)

# some changed have been done in the function calculate_impedance and convert_to_directed
protection_data= simulate_faults_for_all_lines(net, Protection_devices_fix)

# Convert the list of dictionaries into a DataFrame
protection_df = pd.DataFrame(protection_data)

# write to Excel file
protection_df.to_excel('fault_detection_with_new_protection_border_1029.xlsx', index=False)