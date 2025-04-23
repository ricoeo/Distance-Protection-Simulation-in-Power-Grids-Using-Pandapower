import pandas as pd
import numpy as np
from shared.library import *
from shared.protection_border_adjust import *
import matplotlib.pyplot as plt

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# a new create network function is written
net = create_network_withoutparallelline(excel_file)

# initialize the protection device zone settings
Protection_devices = setup_protection_zones(net, excel_file)

# Load the measurement data
excel_file = 'fault_detection_withoutparallel_line.xlsx'
measurement_data = pd.read_excel(excel_file)

# Define a function to adjust zone settings

# Example usage
# for device in Protection_devices:
#     if device == 2:
#         adjust_zone_settings(Protection_devices[device], measurement_data)
adjust_zone = {}
for device in Protection_devices:
    adjust_zone[device] = adjust_zone_boundaries(measurement_data[measurement_data["Device ID"] == device],Protection_devices[device])

# Initialize a list to store the protection data
protection_data = []

# Step 3: Loop through each protection device and extract relevant information
for idx, device in adjust_zone.items():
    protection_data.append({
        'Device ID': device.device_id,                 # ID of the protection device
        'Bus ID': device.bus_id,                       # ID of the associated bus
        'First Line ID': device.associated_line_id,    # ID of the first line connected
        'Replaced Line ID': device.replaced_line_id,   # ID of the replaced line (if applicable)
        'Zone 1 Impedance': device.associated_zone_impedance[0],  # Impedance for Zone 1
        'Zone 2 Impedance': device.associated_zone_impedance[1],  # Impedance for Zone 2
        'Zone 3 Impedance': device.associated_zone_impedance[2]   # Impedance for Zone 3
    })

# Step 4: Convert the list of protection data into a DataFrame
protection_df = pd.DataFrame(protection_data)

# Step 5: Save the DataFrame to an Excel file
output_file = 'adjust_zone_setting_1029.xlsx'
protection_df.to_excel(output_file, index=False)

