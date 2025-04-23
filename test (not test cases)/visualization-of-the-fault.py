import matplotlib.pyplot as plt
from shared.library import *

"""
This script is used to visualize the apparent impedance vector in the R-X plane.
An exmaple of the comparison between the real fault-to-device impedance and the apparent impedance 
in case: with wind generation and without wind generation is given.
"""

# Read the data from the Excel file
excel_file = "grid_data_sheet.xlsx"
# define some constant parameters
HV = 110  # High Voltage side in kilovolts
S_base = 100e6  # Base power in watts (100 MW)
S_sc_HV = 5e9  # Short-circuit power at HV side in watts (5 GW)


def create_network_with_timestepdata(excel_file, wind_p, wind_q, load_p, load_q):
    # Select sheets to read
    bus_data = pd.read_excel(excel_file, sheet_name="bus_data", index_col=0)
    load_data = pd.read_excel(excel_file, sheet_name="load_data", index_col=0)
    line_data = pd.read_excel(excel_file, sheet_name="line_data", index_col=0)
    external_grid_data = pd.read_excel(
        excel_file, sheet_name="external_grid_data", index_col=0
    )
    wind_gen_data = pd.read_excel(excel_file, sheet_name="wind_gen_data", index_col=0)

    net = pp.create_empty_network()

    # buses
    for idx in bus_data.index:
        pp.create_bus(
            net,
            vn_kv=bus_data.loc[idx, "vn_kv"],
            name=bus_data.loc[idx, "name"],
            type=bus_data.loc[idx, "type"],
            geodata=tuple(
                map(int, bus_data.loc[idx, "geodata"].strip("()").split(","))
            ),
        )
    # lines
    for idx in line_data.index:
        pp.create_line_from_parameters(
            net,
            from_bus=line_data.loc[idx, "from_bus"],
            to_bus=line_data.loc[idx, "to_bus"],
            length_km=line_data.loc[idx, "length_km"],
            r_ohm_per_km=line_data.loc[idx, "r_ohm_per_km"],
            x_ohm_per_km=line_data.loc[idx, "x_ohm_per_km"],
            c_nf_per_km=line_data.loc[idx, "c_nf_per_km"],
            r0_ohm_per_km=line_data.loc[idx, "r0_ohm_per_km"],
            x0_ohm_per_km=line_data.loc[idx, "x0_ohm_per_km"],
            c0_nf_per_km=line_data.loc[idx, "c0_nf_per_km"],
            max_i_ka=line_data.loc[idx, "max_i_ka"],
            parallel=line_data.loc[idx, "parallel"],
        )

    # loads
    for idx in load_data.index:
        pp.create_load(
            net,
            bus=load_data.at[idx, "bus"],
            p_mw=load_p.at[idx],
            q_mvar=load_q.at[idx],
            name=load_data.at[idx, "name"],
        )
    # external grids
    for idx in external_grid_data.index:
        pp.create_ext_grid(
            net,
            bus=external_grid_data.at[idx, "bus"],
            vm_pu=external_grid_data.at[idx, "vm_pu"],
            va_degree=external_grid_data.at[idx, "va_degree"],
            name=external_grid_data.at[idx, "name"],
            s_sc_max_mva=5e3,
            rx_max=0.1,
        )

    # generators
    for idx in wind_gen_data.index:
        # if idx == 0:
        #     continue
        pp.create_sgen(
            net,
            bus=wind_gen_data.at[idx, "bus"],
            p_mw=wind_p.at[idx],
            q_mvar=wind_q.at[idx],
            sn_mva=wind_gen_data.at[idx, "sn_mva"],
            name=wind_gen_data.at[idx, "name"],
            generator_type="current_source",
            k=1.2,
            kappa=1.2,
        )

    print(net)
    return net


# Usage
# 353
# wind_p1 = pd.Series(
#     [
#         0.300262006,
#         0.878456093,
#         0.126953949,
#         0.400349341,
#         0.200174671,
#         1.000873354,
#         0.018136278,
#     ]
# )
# 476
# wind_p2 = pd.Series(
#     [
#         8.653514251,
#         3.679389946,
#         6.534830395,
#         11.538019,
#         5.769009501,
#         28.8450475,
#         0.933547199,
#     ]
# )
# 93
wind_p1 = pd.Series(
    [
        1.483363464,
        0.622553816,
        0.700611348,
        1.977817951,
        0.988908976,
        4.944544878,
        0.100087335,
    ]
)
# 291
wind_p2 = pd.Series(
    [
        1.674608665,
        1.192890534,
        0.622553816,
        2.232811553,
        1.116405777,
        5.582028883,
        0.088936259,
    ]
)
wind_q1 = 0.3 * wind_p1
wind_q2 = 0.3 * wind_p2
# 353
# load_p1 = pd.Series([228.5786613, 228.5786613, 114.2893306])
# load_q1 = pd.Series([-2.697479198, -2.697479198, -1.348739599])

# load_p2 = pd.Series([228.5786613, 228.5786613, 114.2893306])
# load_q2 = pd.Series([-2.697479198, -2.697479198, -1.348739599])

# 93
load_p1 = pd.Series([314.4821634, 314.4821634, 157.2410817])
load_q1 = pd.Series([1.394948903, 1.394948903, 0.697474452])

# 291
load_p2 = pd.Series([153.9507101, 153.9507101, 76.97535507])
load_q2 = pd.Series([-2.964443182, -2.964443182, -1.482221591])

net_loadhigh = create_network_with_timestepdata(
    excel_file, wind_p1, wind_q1, load_p1, load_q1
)
net_loadlow = create_network_with_timestepdata(
    excel_file, wind_p2, wind_q2, load_p2, load_q2
)

# the grid topology is not changed so the protection zone setting should be the same for both cases
Protection_devices = setup_protection_zones(net_loadhigh, excel_file)

# Call the function to simulate faults and get protection data
protection_data_1 = simulate_faults_for_all_lines(net_loadhigh, Protection_devices)
# protection_df_1 = pd.DataFrame(protection_data_1)
# protection_df_1.to_excel("protection_data_1.xlsx")
protection_data_2 = simulate_faults_for_all_lines(net_loadlow, Protection_devices)

# Convert the list of dictionaries into a DataFrame
protection_df_1 = pd.DataFrame(protection_data_1)
protection_df_2 = pd.DataFrame(protection_data_2)
protection_df_1.to_excel("protection_data_1.xlsx")
protection_df_2.to_excel("protection_data_2.xlsx")

"""Functions for simple plotting"""


def filter_device_zone_polygon(zone_index, device_index, r_arc, protection_device_list):
    Protection_device_zone = protection_device_list[
        device_index
    ].associated_zone_impedance[zone_index]
    Protection_device_polygon = Polygon(
        [
            (0, 0),
            (
                -Protection_device_zone.imag * math.tan(math.radians(30)),
                Protection_device_zone.imag,
            ),
            (Protection_device_zone.real + r_arc, Protection_device_zone.imag),
            (
                Protection_device_zone.real + r_arc,
                -(Protection_device_zone.real + r_arc) * math.tan(math.radians(22)),
            ),
        ]
    )
    return Protection_device_polygon


def filter_referred_point(df, device_id, fault_line_id, distance_from_bus):
    filtered_df = df.loc[
        (df["Device ID"] == device_id)
        & (df["Fault_line_id"] == fault_line_id)
        & (df["Distance_from_bus"] == distance_from_bus)
    ]
    # Extract the Impedance_calculated value
    if not filtered_df.empty:  # Check if the filtered DataFrame is not empty
        impedance_value = filtered_df["Impedance_calculated"].values[0]
    else:
        print(
            "the simulation result is not generated or the fault simulation case is invalid"
        )
        return None  # Handle the case when no matching row is found
    impedance_point = Point(impedance_value.real, impedance_value.imag)
    return impedance_point


def filter_sensed_point(df, device_id, fault_line_id, distance_from_bus):
    filtered_df = df.loc[
        (df["Device ID"] == device_id)
        & (df["Fault_line_id"] == fault_line_id)
        & (df["Distance_from_bus"] == distance_from_bus)
    ]
    # Extract the Impedance_calculated value
    if not filtered_df.empty:  # Check if the filtered DataFrame is not empty
        r_value = filtered_df["r_sensed"].values[0]
        angle_value = filtered_df["angle_sensed"].values[0]
    else:
        print(
            "the simulation result is not generated or the fault simulation case is invalid"
        )
        return None  # Handle the case when no matching row is found

    sensed_point = Point(
        r_value * math.cos(angle_value * math.pi / 180),
        r_value * math.sin(angle_value * math.pi / 180),
    )
    return sensed_point


# case 1: after no wind generation,the same device before and after, compare with supposed to be
# device 14
# Define constants and parameters for the plot
r_arc_2_5 = 0
device_id = 2
fault_line_id = 3
distance_from_bus = 0.50

# Retrieve polygons for Zone 1 and Zone 2
Protect_14_zone1_polygon = filter_device_zone_polygon(
    0, device_id, r_arc_2_5, Protection_devices
)
Protect_14_zone2_polygon = filter_device_zone_polygon(
    1, device_id, r_arc_2_5, Protection_devices
)

# Retrieve the referred and sensed points for both cases (with and without wind generation)
with_gen_referred = filter_referred_point(
    protection_df_1, device_id, fault_line_id, distance_from_bus
)
with_gen_sensed = filter_sensed_point(
    protection_df_1, device_id, fault_line_id, distance_from_bus
)
without_gen_referred = filter_referred_point(
    protection_df_2, device_id, fault_line_id, distance_from_bus
)
without_gen_sensed = filter_sensed_point(
    protection_df_2, device_id, fault_line_id, distance_from_bus
)


# Function to plot the polygon
def plot_polygon(polygon, color, label):
    xx, y = polygon.exterior.xy
    plt.plot(xx, y, color=color, label=label)
    return min(xx), max(xx), min(y), max(y)


# Function to plot points with coordinates as labels
def plot_point(point, label, marker, text_color):
    plt.plot(
        point.x, point.y, marker, label=label
    )  # Use marker string directly, without color argument
    plt.text(
        point.x + 0.1,
        point.y - 0.1,
        f"({point.x:.3f}, {point.y:.3f})",
        fontsize=12,
        color=text_color,
    )


# Set up figure
plt.figure(figsize=(10, 6))

# Plot the polygons for Zone 1 and Zone 2
x_min_1, x_max_1, y_min_1, y_max_1 = plot_polygon(
    Protect_14_zone1_polygon, "black", "Zone 1 polygon"
)
x_min_2, x_max_2, y_min_2, y_max_2 = plot_polygon(
    Protect_14_zone2_polygon, "gray", "Zone 2 polygon"
)

# Calculate dynamic limits based on the polygons
x_min, x_max = min(x_min_1, x_min_2), max(x_max_1, x_max_2)
y_min, y_max = min(y_min_1, y_min_2), max(y_max_1, y_max_2)

# Plot referred and sensed points (with and without wind generation)
plot_point(with_gen_referred, "case_14_with_gen_referred", "ro", "gray")
plot_point(with_gen_sensed, "case_14_with_gen_sensed", "bo", "gray")
plot_point(without_gen_referred, "case_14_without_gen_referred", "r+", "gray")
plot_point(without_gen_sensed, "case_14_without_gen_sensed", "b+", "gray")

# Add arrows and labels for X and R axes
arrow_length_x = (x_max - x_min) * 1.2
arrow_length_y = (y_max - y_min) * 1.2
head_width = (y_max - y_min) * 0.05
head_length = (x_max - x_min) * 0.05

plt.arrow(
    x_min,
    0,
    arrow_length_x,
    0,
    head_width=head_width,
    head_length=head_length,
    fc="black",
    ec="black",
)
plt.arrow(
    0,
    y_min,
    0,
    arrow_length_y,
    head_width=head_width,
    head_length=head_length,
    fc="black",
    ec="black",
)
plt.text(
    x_min + arrow_length_x + head_length + 0.8,
    0,
    "R",
    fontsize=12,
    ha="center",
    va="center",
)
plt.text(
    0,
    y_min + arrow_length_y + head_length + 0.8,
    "X",
    fontsize=12,
    ha="center",
    va="center",
)

# Adjust plot settings
plt.grid(True)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.xticks([])
plt.yticks([])

# Add legend and show plot
plt.legend(loc="upper right")
plt.show()
