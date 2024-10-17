from shared.library import *

# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

net_with_gen = create_network(excel_file)
# net_without_gen = create_network_without_gen(excel_file)

#test different arc resistance configuration
min_r_arc = 0.5  # Minimum arc resistance
max_r_arc = 10   # Maximum arc resistance
step_size = 0.5 

r_arc = min_r_arc

# Generate R_arc values and perform simulation for each one
while r_arc <= max_r_arc:
    # Setup protection zones using the current R_arc value
    # don't run this line before changing the initalize method of protection device
    Protection_devices = setup_protection_zones(net_with_gen, excel_file, r_arc)
    
    # Simulate faults for all lines with the current Protection_devices
    protection_data_with_gen = simulate_faults_for_all_lines(net_with_gen, Protection_devices)
    
    # Convert the protection data to a DataFrame
    protection_df_with_gen = pd.DataFrame(protection_data_with_gen)
    
    # Define a dynamic Excel file name using the current R_arc value
    excel_file_name = f'fault_detection_check_r_arc_{r_arc:.1f}.xlsx'
    
    # Save the DataFrame to the Excel file
    protection_df_with_gen.to_excel(excel_file_name, index=False)
    
    # Increment the R_arc value by the step size
    r_arc += step_size

