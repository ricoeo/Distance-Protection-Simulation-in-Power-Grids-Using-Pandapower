from shared.library import *
from shared.extract_profile import *
# Read the data from the Excel file
excel_file = 'grid_data_sheet.xlsx'

# Step 1: Create the network with generation
net_with_gen = create_network(excel_file)

# Step 2: load the profiles prepared in the extract profile header
net_with_dynamic_profile = load_profiles(net_with_gen)

# Step 3: run the timeseries calculations
ts.run_timeseries(net_with_dynamic_profile)

print(net_with_dynamic_profile)