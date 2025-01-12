from shared.library import *
from shared.protection_border_adjust import (
    adjust_protection_zone_with_measurement_data as adjust_protection_zone,
)

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

# a new create network function is written
net = create_network_without_BE(excel_file)

# import the previous measurement data, always check it to the up-to-date measurement data
excel_file_measure = "fault_detection_check_BE.xlsx"
measurement_data = pd.read_excel(excel_file_measure)

# change the protection zone setting according to the measurement data
Protection_devices = setup_protection_zones(net, excel_file)
Protection_devices_fix = adjust_protection_zone(measurement_data, Protection_devices)

# simulate faults
protection_data = simulate_faults_for_all_lines(net, Protection_devices_fix)

# Convert the list of dictionaries into a DataFrame
protection_df = pd.DataFrame(protection_data)

# write to Excel file
protection_df.to_excel(
    "fault_detection_with_new_protection_border_BE.xlsx", index=False
)
