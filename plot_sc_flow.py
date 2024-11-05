import pandapower as pp
import pandas as pd
import pandapower.plotting as pplot
from pandapower.plotting import simple_plotly
from pandapower.plotting import simple_plot
from pandapower.plotting.collections import *
import math
import matplotlib.pyplot as plt
import numpy as np
import pandapower.shortcircuit as sc

# Read the data from the excel file 
excel_file = 'grid_data_sheet.xlsx'

# Select sheets to read
bus_data = pd.read_excel(excel_file, sheet_name='bus_data',index_col=0)
load_data = pd.read_excel(excel_file, sheet_name='load_data',index_col=0)
line_data = pd.read_excel(excel_file, sheet_name='line_data',index_col=0)
external_grid_data = pd.read_excel(excel_file, sheet_name='external_grid_data',index_col=0)
wind_gen_data = pd.read_excel(excel_file, sheet_name='wind_gen_data',index_col=0)

#create network as topology shows
net = pp.create_empty_network()
#buses
for idx in bus_data.index:
    pp.create_bus(net,vn_kv= bus_data.at[idx, "vn_kv"],name= bus_data.at[idx, "name"],type =bus_data.at[idx, "type"],geodata=tuple(map(int, bus_data.at[idx, "geodata"].strip('()').split(','))))
#lines
for idx in line_data.index:
    pp.create_line_from_parameters(net,from_bus=line_data.at[idx, "from_bus"],to_bus=line_data.at[idx, "to_bus"],length_km=line_data.at[idx, "length_km"],r_ohm_per_km=line_data.at[idx, "r_ohm_per_km"],x_ohm_per_km=line_data.at[idx, "x_ohm_per_km"],c_nf_per_km=line_data.at[idx, "c_nf_per_km"],r0_ohm_per_km=line_data.at[idx, "r0_ohm_per_km"],x0_ohm_per_km=line_data.at[idx, "x0_ohm_per_km"],c0_nf_per_km=line_data.at[idx, "c0_nf_per_km"],max_i_ka=line_data.at[idx, "max_i_ka"],parallel=line_data.at[idx, "parallel"])

net.line.at[1, 'in_service'] = False
net.line.at[0., 'length_km'] -net.line.at[0., 'length_km'] * 0.5
#loads
for idx in load_data.index:
    pp.create_load(net,bus=load_data.at[idx, "bus"],p_mw=load_data.at[idx, "p_mw"],q_mvar=load_data.at[idx, "q_mvar"],name=load_data.at[idx, "name"])
#external grids 
for idx in external_grid_data.index:
    pp.create_ext_grid(net,bus=external_grid_data.at[idx, "bus"],vm_pu=external_grid_data.at[idx, "vm_pu"],va_degree=external_grid_data.at[idx, "va_degree"],name=external_grid_data.at[idx, "name"],s_sc_max_mva=5e9,rx_max=0.1)            
    
#generators
for idx in wind_gen_data.index:
    pp.create_sgen(net,bus=wind_gen_data.at[idx, "bus"],p_mw=wind_gen_data.at[idx, "p_mw"],q_mvar=wind_gen_data.at[idx, "q_mvar"],sn_mva=wind_gen_data.at[idx, "sn_mva"],name=wind_gen_data.at[idx, "name"],k=1.2)

pp.runpp(net)
sc.calc_sc(net,fault= '3ph',bus=4, branch_results = True)


def create_voltage_dict(net):
    # Initialize an empty dictionary to store bus voltages
    bus_voltage_dict = {}
    # Iterate through the lines in res_line_sc
    for idx in net.res_line_sc.index:
        # Get bus IDs for the line from the net.line DataFrame using idx
        if net.line.at[idx, 'in_service']:
            from_bus = net.line.loc[idx, "from_bus"]
            to_bus = net.line.loc[idx, "to_bus"]

            # Get the voltage from the results
            vm_from = net.res_line_sc.loc[idx, "vm_from_pu"]
            vm_to = net.res_line_sc.loc[idx, "vm_to_pu"]

            # Update the dictionary with the voltages
            bus_voltage_dict[from_bus] = vm_from
            bus_voltage_dict[to_bus] = vm_to

    return bus_voltage_dict

def create_current_dict(net):
    """
    Create a dictionary containing current magnitudes and their directions
    based on voltage and current angles from the pandapower network.

    Parameters:
        net: pandapower network

    Returns:
        current_dict: Dictionary with bus_id as key and 
                       a dictionary containing current magnitude and direction as value.
    """
    current_dict = {}

    for idx, row in net.res_line_sc.iterrows():
        if net.line.at[idx, 'in_service']:
            current_and_direction_dict = {}
            # Get the current magnitude
            ikss = row["ikss_ka"]
            current_and_direction_dict['current'] = ikss
            
            # Get the direction of the power flow 
            from_bus = net.line.loc[idx, "from_bus"]
            to_bus = net.line.loc[idx, "to_bus"]

            # Get voltage angles
            voltage_angle_from = net.res_line_sc.loc[idx, "va_from_degree"]

            # get current angle
            current_angle_to = net.res_line_sc.loc[idx, "ikss_to_degree"]   
            
            direction = (from_bus, to_bus)
            # Determine the direction of current flow
            if math.cos((voltage_angle_from-current_angle_to) * math.pi / 180) > 0:
                direction = (from_bus, to_bus)  # Current flows from from_bus to to_bus
            else:
                direction = (to_bus, from_bus)  # Current flows from to_bus to from_bus

            # Create dictionary entries for both buses
            current_and_direction_dict['direction'] = direction
            current_dict[idx] = current_and_direction_dict

    return current_dict

def my_simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0,
                trafo_size=1.0, plot_loads=False, plot_gens=False, plot_sgens=False, sgn_oritation = 0,load_size=1.0, gen_size=1.0, sgen_size=1.0,
                switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True,
                bus_color='b', line_color='grey',  dcline_color='c', trafo_color='k',
                ext_grid_color='y', switch_color='k', library='igraph', show_plot=True, ax=None):
    """
        inherit from the simple plot function but add the component direction input parameter
    """
    # don't hide lines if switches are plotted
    if plot_line_switches:
        respect_switches = False

    # create geocoord if none are available
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        pplot.create_generic_coordinates(net, respect_switches=respect_switches, library=library)

    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        sizes = pplot.get_collection_sizes(net, bus_size, ext_grid_size, trafo_size,
                                     load_size, sgen_size, switch_size, switch_distance, gen_size)
        bus_size = sizes["bus"]
        ext_grid_size = sizes["ext_grid"]
        trafo_size = sizes["trafo"]
        sgen_size = sizes["sgen"]
        load_size = sizes["load"]
        switch_size = sizes["switch"]
        switch_distance = sizes["switch_distance"]
        gen_size = sizes["gen"]

    # create bus collections to plot
    bc = create_bus_collection(net, net.bus.index, size=bus_size, color=bus_color, zorder=10)

    # if bus geodata is available, but no line geodata
    use_bus_geodata = len(net.line_geodata) == 0
    in_service_lines = net.line[net.line.in_service].index
    nogolines = set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)]) \
        if respect_switches else set()
    plot_lines = in_service_lines.difference(nogolines)
    plot_dclines = net.dcline.in_service

    # create line collections
    lc = create_line_collection(net, plot_lines, color=line_color, linewidths=line_width,
                                use_bus_geodata=use_bus_geodata)
    collections = [bc, lc]

    # create dcline collections
    if len(net.dcline) > 0:
        dclc = create_dcline_collection(net, plot_dclines, color=dcline_color,
                                        linewidths=line_width)
        collections.append(dclc)

    # create ext_grid collections
    # eg_buses_with_geo_coordinates = set(net.ext_grid.bus.values) & set(net.bus_geodata.index)
    if len(net.ext_grid) > 0:
        sc = create_ext_grid_collection(net, size=ext_grid_size, orientation=0,
                                        ext_grids=net.ext_grid.index, patch_edgecolor=ext_grid_color,
                                        zorder=11)
        collections.append(sc)

    # create trafo collection if trafo is available
    trafo_buses_with_geo_coordinates = [t for t, trafo in net.trafo.iterrows()
                                        if trafo.hv_bus in net.bus_geodata.index and
                                        trafo.lv_bus in net.bus_geodata.index]
    if len(trafo_buses_with_geo_coordinates) > 0:
        tc = create_trafo_collection(net, trafo_buses_with_geo_coordinates,
                                     color=trafo_color, size=trafo_size)
        collections.append(tc)

    # create trafo3w collection if trafo3w is available
    trafo3w_buses_with_geo_coordinates = [
        t for t, trafo3w in net.trafo3w.iterrows() if trafo3w.hv_bus in net.bus_geodata.index and
                                                      trafo3w.mv_bus in net.bus_geodata.index and trafo3w.lv_bus in net.bus_geodata.index]
    if len(trafo3w_buses_with_geo_coordinates) > 0:
        tc = create_trafo3w_collection(net, trafo3w_buses_with_geo_coordinates,
                                       color=trafo_color)
        collections.append(tc)

    if plot_line_switches and len(net.switch):
        sc = create_line_switch_collection(
            net, size=switch_size, distance_to_bus=switch_distance,
            use_line_geodata=not use_bus_geodata, zorder=12, color=switch_color)
        collections.append(sc)

    if plot_sgens and len(net.sgen):
        sgc = create_sgen_collection(net, size=sgen_size, orientation=sgn_oritation)
        collections.append(sgc)
    if plot_gens and len(net.gen):
        gc = create_gen_collection(net, size=gen_size)
        collections.append(gc)
    if plot_loads and len(net.load):
        lc = create_load_collection(net, size=load_size)
        collections.append(lc)

    if len(net.switch):
        bsc = create_bus_bus_switch_collection(net, size=switch_size)
        collections.append(bsc)

    ax = draw_collections(collections, ax=ax)
    if show_plot:
        if not MATPLOTLIB_INSTALLED:
            soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
        plt.show()
    return ax

def plot_short_circuit_results(net):
    if net.res_bus_sc.empty or net.res_line_sc.empty:
        raise ValueError("Short circuit results are missing. Please run the short circuit simulation before plotting.")
    voltage_dict = create_voltage_dict(net)
    current_dict = create_current_dict(net)
    
    # Create color maps
    voltage_cmap = plt.get_cmap("Blues")  # Light blue to dark blue
    # current_cmap = plt.get_cmap("cividis")  # Light yellow to dark yellow, not used anymore

    # Normalize voltage data for color mapping
    min_voltage = min(voltage_dict.values())
    max_voltage = max(voltage_dict.values())
    voltage_norm = plt.Normalize(vmin=min_voltage, vmax=max_voltage)

    # Normalize current data for color mapping
    current_values = [info['current'] for info in current_dict.values()]
    min_current = min(current_values)
    max_current = max(current_values)
    current_norm = plt.Normalize(vmin=min_current, vmax=max_current)
    
    # Base plot with pandapower simple_plot
    ax = my_simple_plot(net,plot_loads=True,scale_size =True,load_size  =2, plot_sgens =True, sgen_size=2,bus_color = '#00000000', show_plot = False, sgn_oritation=np.pi/2)

    # Plot bus voltage with darker color means higher voltage
    for bus_id, voltage in voltage_dict.items():
        coords = net.bus_geodata.loc[bus_id]
        color = voltage_cmap(voltage_norm(voltage))
        ax.scatter(coords.x, coords.y, color=color, edgecolor='black', s=100, 
                label=f'Bus {bus_id}: {voltage:.2f} pu', zorder=5) 
        
    # Plot current magnitude and direction
    for _, info in current_dict.items():
        direction = info['direction']
        from_bus, to_bus = direction

        # Get coordinates for the from and to buses
        from_coords = net.bus_geodata.loc[from_bus]
        to_coords = net.bus_geodata.loc[to_bus]
        midpoint_x = (from_coords.x + to_coords.x) / 2
        midpoint_y = (from_coords.y + to_coords.y) / 2
        # Color is difficult to visulize, change it to thickness instead
        current_magnitude = info['current']
        line_thickness = (current_magnitude/max_current + 1)*0.04
        ax.text(midpoint_x, midpoint_y, f"{current_magnitude:.2f} kA", color="black", ha='center', va='center', fontsize=8,fontweight = 600, zorder=4)

        # Calculate arrow direction and apply a scale factor to make it more readable
        dx, dy = to_coords.x - from_coords.x, to_coords.y - from_coords.y
        length_factor = 0.85  # Adjust length of arrows to improve readability
        dx, dy = dx * length_factor, dy * length_factor

        # Offset for starting point to make the head clearly visible
        offset_factor = 0.05
        offset_x, offset_y = dx * offset_factor, dy * offset_factor

        # Draw the improved arrow with larger head and offset
        ax.arrow(from_coords.x + offset_x, from_coords.y + offset_y, dx, dy,
                 color=color, width=line_thickness, head_width=line_thickness*4, head_length=line_thickness*4,
                 length_includes_head=True, zorder=3 )

    # Show plot with legend
    ax.legend()
    plt.show()

# Usage
plot_short_circuit_results(net)