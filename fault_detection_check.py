from shared.library import *

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

net = create_network(excel_file)

# set the protection zone setting
Protection_devices = setup_protection_zones(net, excel_file)

# Call the function to simulate faults and get protection data
protection_data = simulate_faults_for_all_lines(net, Protection_devices)

# Convert the list of dictionaries into a DataFrame
protection_df = pd.DataFrame(protection_data)

# write to Excel file
protection_df.to_excel("fault_detection_check_0113.xlsx", index=False)
