from shared.library import *


# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"
excel_file_gen = "./timeseries_results/res_sgen/p_mw.xlsx"

# Load the generator power data (assume it's structured with rows for each time step)
gen_data = pd.read_excel(excel_file_gen, index_col=0)

# Limit the number of rows to process
MAX_ROWS = 30  # Set the limit here
gen_data = gen_data.head(MAX_ROWS)  # Process only the first MAX_ROWS rows

"""start: check the fault bus current"""
output_excel = "fault_bus_current.xlsx"
# Initialize a list to store results
results = []

# Iterate over generator data
for i, row in gen_data.iterrows():
    # Create the network for the current time step
    net = create_network_without_BE_AB1_reference(excel_file, i, excel_file_gen)

    # Define line and fault location
    line = net.line.loc[0]  # Assuming line[0] is the target line
    fault_location = line.length_km / 2

    # Create a temporary bus and split the line into two parts
    temp_bus = pp.create_bus(
        net,
        vn_kv=HV,
        type="n",
        name="fault_bus",
    )

    temp_line_part1 = pp.create_line_from_parameters(
        net,
        from_bus=line.from_bus,
        to_bus=temp_bus,
        length_km=fault_location,
        in_service=True,
        **{
            attr: line[attr]
            for attr in line.index
            if attr
            not in [
                "from_bus",
                "to_bus",
                "length_km",
                "name",
                "std_type",
                "in_service",
            ]
        },
    )

    temp_line_part2 = pp.create_line_from_parameters(
        net,
        from_bus=temp_bus,
        to_bus=line.to_bus,
        length_km=line.length_km - fault_location,
        in_service=True,
        **{
            attr: line[attr]
            for attr in line.index
            if attr
            not in [
                "from_bus",
                "to_bus",
                "length_km",
                "name",
                "std_type",
                "in_service",
            ]
        },
    )

    # Simulate a single-phase short circuit at the temporary bus
    sc.calc_sc(
        net,
        fault="1ph",
        bus=temp_bus,
        branch_results=True,
        return_all_currents=True,
    )

    # Extract the short-circuit current at the fault bus
    fault_bus_current = net.res_bus_sc.loc[temp_bus, "ikss_ka"]

    fault_line_9_current = net.res_line_sc["ikss_ka"].iloc[9]

    # Append results to the list
    results.append(
        {
            "Time Step": i,
            "Fault Bus Current (kA)": fault_bus_current,
            "line 9 Current (kA)": fault_line_9_current,
        }
    )
# Convert results to a DataFrame
results_df = pd.DataFrame(results)
# Write the results to an Excel file
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Fault Bus Currents")
