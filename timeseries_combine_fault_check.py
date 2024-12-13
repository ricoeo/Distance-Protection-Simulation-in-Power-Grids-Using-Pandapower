from shared.library import *
from shared.extract_profile import *
from shared.protection_border_adjust import (
    adjust_protection_zone_with_measurement_data as adjust_protection_zone,
)

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

net = create_network_without_BE_AB1(excel_file)


# import the previous measurement data, always check it to the up-to-date measurement data
excel_file_measure = "fault_detection_check_BE_AB1.xlsx"
measurement_data = pd.read_excel(excel_file_measure)

# change the protection zone setting according to the measurement data
Protection_devices = setup_protection_zones(net, excel_file)
Protection_devices = adjust_protection_zone(measurement_data, Protection_devices)

# load the profiles prepared in the extract profile header
net_with_dynamic_profile = load_profiles(net)

# Add the custom fault simulation controller
fault_sim_controller = FaultSimulationController(
    net_with_dynamic_profile, Protection_devices
)


# specify the result save folder
output_dir = "timeseries_results_optimal_border_0912"
os.makedirs(output_dir, exist_ok=True)

# create output writer
ow = ts.OutputWriter(
    net_with_dynamic_profile, output_path=output_dir, output_file_type=".xlsx"
)
ow.log_variable("res_load", "p_mw")
# ow.log_variable("res_load", "q_mvar")
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("custom", "Value")

# the default logger is disabled
ow.remove_log_variable("res_line", "loading_percent")
ow.remove_log_variable("res_bus", "vm_pu")


# run the timeseries calculations, it will take quite long so adjusting the time_steps as need
ts.run_timeseries(
    net_with_dynamic_profile,
    time_steps=list(range(0, 20)),
    continue_on_divergence=True,
)
