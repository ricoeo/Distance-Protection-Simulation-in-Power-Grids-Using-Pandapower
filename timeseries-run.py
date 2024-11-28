from shared.library import *
from shared.extract_profile import *

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

# Step 1: Create the network with generation
net_with_gen = create_network_without_BE_AB1(excel_file)

# Step 2: load the profiles prepared in the extract profile header
net_with_dynamic_profile = load_profiles(net_with_gen)

# this is how you add customized result into the output writer
# Add a custom table to the network
net_with_dynamic_profile.custom_results = pd.DataFrame(
    index=net_with_dynamic_profile.bus.index
)
net_with_dynamic_profile.custom_results["custom_metric"] = 0.0

# create output writer

ow = ts.OutputWriter(
    net_with_dynamic_profile, output_path="./", output_file_type=".xlsx"
)
ow.log_variable("res_load", "p_mw")
ow.log_variable("res_load", "q_mvar")
ow.log_variable("res_sgen", "p_mw")
ow.remove_log_variable("res_line", "loading_percent")
ow.remove_log_variable("res_bus", "vm_pu")


# Step 3: run the timeseries calculations
ts.run_timeseries(net_with_dynamic_profile, continue_on_divergence=True)
