from shared.library import *


# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"
excel_file_gen = "./timeseries_results/res_sgen/p_mw.xlsx"

# Load the generator power data (assume it's structured with rows for each time step)
gen_data = pd.read_excel(excel_file_gen, index_col=0)

# Limit the number of rows to process
MAX_ROWS = 10  # Set the limit here
gen_data = gen_data.head(MAX_ROWS)  # Process only the first MAX_ROWS rows


net_temp = create_network_without_BE_AB1(excel_file)

# Set the protection zone setting
Protection_devices = setup_protection_zones(net_temp, excel_file)


# Create a writer for saving results to Excel with multiple sheets
output_file = "fault_detection_check_results_2.xlsx"


with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    # Iterate through each row in the generator data
    for i, row in gen_data.iterrows():
        # Create the network
        net = create_network_without_BE_AB1_reference(excel_file, i, excel_file_gen)

        # Simulate faults and get protection data
        protection_data = simulate_faults_for_all_lines(net, Protection_devices)

        # Convert the list of dictionaries into a DataFrame
        protection_df = pd.DataFrame(protection_data)

        # Write the DataFrame to a separate sheet
        protection_df.to_excel(writer, sheet_name=f"Scenario_{i}", index=False)

# Confirm completion
print(f"Results saved to {output_file}")
