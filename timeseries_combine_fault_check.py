from shared.library import *
from shared.extract_profile import *

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

net = create_network_without_BE_AB1(excel_file)

# set the protection zone setting
Protection_devices = setup_protection_zones(net, excel_file)

# load the profiles prepared in the extract profile header
net_with_dynamic_profile = load_profiles(net)

net_with_dynamic_profile.custom = simulate_faults_for_all_lines(
    net_with_dynamic_profile, Protection_devices, show_sum_up_information_flag=True
)
# specify the result save folder
output_dir = "timeseries_results"
os.makedirs(output_dir, exist_ok=True)

# create output writer

# run the timeseries calculations
ts.run_timeseries(net_with_dynamic_profile, continue_on_divergence=True)
