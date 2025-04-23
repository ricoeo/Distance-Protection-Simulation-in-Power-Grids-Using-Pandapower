from shared.library import *

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

# Step 1: Create the network with generation
net = create_network(excel_file)
# Step 2: Set up protection zones for the network
Protection_devices = setup_protection_zones(net, excel_file)

# Initialize a list to store the protection data
protection_data = []

# Step 3: Loop through each protection device and extract relevant information
for idx, device in Protection_devices.items():
    protection_data.append(
        {
            "Device ID": device.device_id,  # ID of the protection device
            "Bus ID": device.bus_id,  # ID of the associated bus
            "First Line ID": device.associated_line_id,  # ID of the first line connected
            "Replaced Line ID": device.replaced_line_id,  # ID of the replaced line (if applicable)
            "Zone 1 Impedance": device.associated_zone_impedance[
                0
            ],  # Impedance for Zone 1
            "Zone 2 Impedance": device.associated_zone_impedance[
                1
            ],  # Impedance for Zone 2
            "Zone 3 Impedance": device.associated_zone_impedance[
                2
            ],  # Impedance for Zone 3
        }
    )

# Step 4: Convert the list of protection data into a DataFrame
protection_df = pd.DataFrame(protection_data)

# Step 5: Save the DataFrame to an Excel file
output_file = "protection_zones_0312.xlsx"
protection_df.to_excel(output_file, index=False)

# Print a confirmation message
print(f"Protection zone data saved to {output_file}")
