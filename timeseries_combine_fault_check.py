from shared.library import *
from shared.extract_profile import *
from shared.protection_border_adjust import (
    adjust_protection_zone_with_measurement_data as adjust_protection_zone,
)

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"

net = create_network(excel_file)

Protection_devices = setup_protection_zones(net, excel_file)
# # import the previous measurement data, always check it to the up-to-date measurement data
# excel_file_measure = "fault_detection_check_BE_AB1.xlsx"
# measurement_data = pd.read_excel(excel_file_measure)

# # change the protection zone setting according to the measurement data
# Protection_devices = setup_protection_zones(net, excel_file)
# Protection_devices = adjust_protection_zone(measurement_data, Protection_devices)

# load the profiles prepared in the extract profile header
net_with_dynamic_profile = load_profiles(net)

# Add the custom fault simulation controller
fault_sim_controller = FaultSimulationController(
    net_with_dynamic_profile, Protection_devices
)


# specify the base result save folder
base_output_dir = "./timeseries_results"

# create base output directory if it does not exist
os.makedirs(base_output_dir, exist_ok=True)

# run the timeseries calculations, save results periodically
ONE_YEAR_DATAPOINT = 365 * 24 * 6
time_steps = list(range(0, ONE_YEAR_DATAPOINT))
save_interval = 100  # save results after every 100 steps
folder_index = math.ceil(ONE_YEAR_DATAPOINT * 1.0 / save_interval)

for i in range(0, folder_index):
    # Create a new folder for each batch of results
    if i < 32:
        continue
    output_dir = os.path.join(base_output_dir, f"results_{i}")
    os.makedirs(output_dir, exist_ok=True)

    # Create output writer for each batch
    ow = ts.OutputWriter(
        net_with_dynamic_profile, output_path=output_dir, output_file_type=".xlsx"
    )
    ow.log_variable("res_load", "p_mw")
    ow.log_variable("res_load", "q_mvar")
    ow.log_variable("res_sgen", "p_mw")
    ow.log_variable("custom", "Value")

    # Disable the default logger
    ow.remove_log_variable("res_line", "loading_percent")
    ow.remove_log_variable("res_bus", "vm_pu")

    if i != folder_index - 1:
        # Run timeseries calculations for the current batch
        ts.run_timeseries(
            net_with_dynamic_profile,
            time_steps=time_steps[i * save_interval : (i + 1) * save_interval],
            continue_on_divergence=True,
        )
    else:
        ts.run_timeseries(
            net_with_dynamic_profile,
            time_steps=time_steps[i * save_interval :],
            continue_on_divergence=True,
        )
