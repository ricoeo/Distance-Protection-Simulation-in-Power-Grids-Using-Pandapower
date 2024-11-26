import numpy as np
import pandapower as pp
import pandas as pd
import pandapower.topology as top
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
import pandapower.shortcircuit as sc
import os
import warnings

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.ERROR)
import pandapower.plotting as pplot
from pandapower.plotting import simple_plot
from pandapower.plotting.collections import *
import matplotlib.pyplot as plt


# define some constant parameters
HV = 110  # High Voltage side in kilovolts
S_base = 100e6  # Base power in watts (100 MW)
S_sc_HV = 5e9  # Short-circuit power at HV side in watts (5 GW)


def create_network(excel_file):
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
            p_mw=load_data.at[idx, "p_mw"],
            q_mvar=load_data.at[idx, "q_mvar"],
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
            s_sc_max_mva=5e9,
            rx_max=0.1,
        )

    # generators
    for idx in wind_gen_data.index:
        # if idx == 0:
        #     continue
        pp.create_sgen(
            net,
            bus=wind_gen_data.at[idx, "bus"],
            p_mw=wind_gen_data.at[idx, "p_mw"],
            q_mvar=wind_gen_data.at[idx, "q_mvar"],
            sn_mva=wind_gen_data.at[idx, "sn_mva"],
            name=wind_gen_data.at[idx, "name"],
            generator_type="current_source",
            k=1.2,
        )

    print(net)
    return net


def create_network_without_gen(excel_file):
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
            vn_kv=bus_data.at[idx, "vn_kv"],
            name=bus_data.at[idx, "name"],
            type=bus_data.at[idx, "type"],
            geodata=tuple(map(int, bus_data.at[idx, "geodata"].strip("()").split(","))),
        )
    # lines
    for idx in line_data.index:
        pp.create_line_from_parameters(
            net,
            from_bus=line_data.at[idx, "from_bus"],
            to_bus=line_data.at[idx, "to_bus"],
            length_km=line_data.at[idx, "length_km"],
            r_ohm_per_km=line_data.at[idx, "r_ohm_per_km"],
            x_ohm_per_km=line_data.at[idx, "x_ohm_per_km"],
            c_nf_per_km=line_data.at[idx, "c_nf_per_km"],
            r0_ohm_per_km=line_data.at[idx, "r0_ohm_per_km"],
            x0_ohm_per_km=line_data.at[idx, "x0_ohm_per_km"],
            c0_nf_per_km=line_data.at[idx, "c0_nf_per_km"],
            max_i_ka=line_data.at[idx, "max_i_ka"],
            parallel=line_data.at[idx, "parallel"],
        )
    # loads
    for idx in load_data.index:
        pp.create_load(
            net,
            bus=load_data.at[idx, "bus"],
            p_mw=load_data.at[idx, "p_mw"],
            q_mvar=load_data.at[idx, "q_mvar"],
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
            s_sc_max_mva=5e9,
            rx_max=0.1,
        )

    # # generators
    # for idx in wind_gen_data.index:
    #     # if idx == 0:
    #     #     continue
    #     pp.create_sgen(net, bus=wind_gen_data.at[idx, "bus"], p_mw=wind_gen_data.at[idx, "p_mw"],
    #                    q_mvar=wind_gen_data.at[idx, "q_mvar"], sn_mva=wind_gen_data.at[idx, "sn_mva"],
    #                    name=wind_gen_data.at[idx, "name"], k=1.2, in_service=False)
    print(net)
    return net


# test the network without parallel line
def create_network_withoutparallelline(excel_file):
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
    net.line.at[1, "in_service"] = False
    net.line.at[0, "length_km"] = net.line.at[0, "length_km"] * 0.5
    # loads
    for idx in load_data.index:
        pp.create_load(
            net,
            bus=load_data.at[idx, "bus"],
            p_mw=load_data.at[idx, "p_mw"],
            q_mvar=load_data.at[idx, "q_mvar"],
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
            s_sc_max_mva=5e9,
            rx_max=0.1,
        )

    # generators
    for idx in wind_gen_data.index:
        # if idx == 0:
        #     continue
        pp.create_sgen(
            net,
            bus=wind_gen_data.at[idx, "bus"],
            p_mw=wind_gen_data.at[idx, "p_mw"],
            q_mvar=wind_gen_data.at[idx, "q_mvar"],
            sn_mva=wind_gen_data.at[idx, "sn_mva"],
            name=wind_gen_data.at[idx, "name"],
            generator_type="current_source",
            k=1.2,
        )

    print(net)
    return net


def create_network_without_BE_AB1(excel_file):
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
    net.line.at[4, "in_service"] = False
    net.line.at[0, "in_service"] = False
    # loads
    for idx in load_data.index:
        pp.create_load(
            net,
            bus=load_data.at[idx, "bus"],
            p_mw=load_data.at[idx, "p_mw"],
            q_mvar=load_data.at[idx, "q_mvar"],
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
            s_sc_max_mva=5e9,
            rx_max=0.1,
        )

    # generators
    for idx in wind_gen_data.index:
        # if idx == 0:
        #     continue
        pp.create_sgen(
            net,
            bus=wind_gen_data.at[idx, "bus"],
            p_mw=wind_gen_data.at[idx, "p_mw"],
            q_mvar=wind_gen_data.at[idx, "q_mvar"],
            sn_mva=wind_gen_data.at[idx, "sn_mva"],
            name=wind_gen_data.at[idx, "name"],
            generator_type="current_source",
            k=1.2,
        )

    print(net)
    return net


def create_network_without_BC(excel_file):
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
    net.line.at[3, "in_service"] = False
    # loads
    for idx in load_data.index:
        pp.create_load(
            net,
            bus=load_data.at[idx, "bus"],
            p_mw=load_data.at[idx, "p_mw"],
            q_mvar=load_data.at[idx, "q_mvar"],
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
            s_sc_max_mva=5e9,
            rx_max=0.1,
        )

    # generators
    for idx in wind_gen_data.index:
        # if idx == 0:
        #     continue
        pp.create_sgen(
            net,
            bus=wind_gen_data.at[idx, "bus"],
            p_mw=wind_gen_data.at[idx, "p_mw"],
            q_mvar=wind_gen_data.at[idx, "q_mvar"],
            sn_mva=wind_gen_data.at[idx, "sn_mva"],
            name=wind_gen_data.at[idx, "name"],
            generator_type="current_source",
            k=1.2,
        )

    print(net)
    return net


def create_network_without_AB1(excel_file):
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
    net.line.at[0, "in_service"] = False
    # loads
    for idx in load_data.index:
        pp.create_load(
            net,
            bus=load_data.at[idx, "bus"],
            p_mw=load_data.at[idx, "p_mw"],
            q_mvar=load_data.at[idx, "q_mvar"],
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
            s_sc_max_mva=5e9,
            rx_max=0.1,
        )

    # generators
    for idx in wind_gen_data.index:
        # if idx == 0:
        #     continue
        pp.create_sgen(
            net,
            bus=wind_gen_data.at[idx, "bus"],
            p_mw=wind_gen_data.at[idx, "p_mw"],
            q_mvar=wind_gen_data.at[idx, "q_mvar"],
            sn_mva=wind_gen_data.at[idx, "sn_mva"],
            name=wind_gen_data.at[idx, "name"],
            generator_type="current_source",
            k=1.2,
        )

    print(net)
    return net


def create_network_meshed_simple(excel_file):
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
    # net.line.at[0, 'length_km'] = net.line.at[0, 'length_km'] * 0.5
    net.line.at[1, "in_service"] = False
    net.line.at[4, "in_service"] = False
    net.line.at[5, "in_service"] = False
    net.line.at[13, "in_service"] = False
    # loads
    for idx in load_data.index:
        pp.create_load(
            net,
            bus=load_data.at[idx, "bus"],
            p_mw=load_data.at[idx, "p_mw"],
            q_mvar=load_data.at[idx, "q_mvar"],
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
            s_sc_max_mva=5e9,
            rx_max=0.1,
        )

    # generators
    for idx in wind_gen_data.index:
        # if idx == 0:
        #     continue
        pp.create_sgen(
            net,
            bus=wind_gen_data.at[idx, "bus"],
            p_mw=wind_gen_data.at[idx, "p_mw"],
            q_mvar=wind_gen_data.at[idx, "q_mvar"],
            sn_mva=wind_gen_data.at[idx, "sn_mva"],
            name=wind_gen_data.at[idx, "name"],
            generator_type="current_source",
            k=1.2,
        )
        # net.sgen.at[idx, 'in_service'] = False
    print(net)
    return net


class ProtectionDevice:
    def __init__(
        self, device_id, bus_id, first_line_id, replaced_line_id, net, depth=3
    ):
        self.device_id = device_id
        self.bus_id = bus_id
        self.associated_line_id = first_line_id
        self.replaced_line_id = replaced_line_id
        self.net = net
        self.associated_zone_impedance = self.find_associated_lines(
            first_line_id, depth
        )
        self.reference_zone_impedance = self.associated_zone_impedance
        # test different r_arc effect
        self.r_arc = 2.5

    def update_associated_line(self):
        if not math.isnan(self.replaced_line_id):
            self.associated_line_id = self.replaced_line_id

    def find_associated_lines(self, start_line_id, depth):
        # Create a graph of the network
        graph = top.create_nxgraph(
            self.net,
            include_lines=True,
            include_impedances=True,
            calc_branch_impedances=True,
            multi=True,
            include_out_of_service=False,
        )
        start_bus = self.bus_id
        # line_impedance=None
        if start_bus == self.net.line.at[start_line_id, "from_bus"]:
            next_bus = self.net.line.at[start_line_id, "to_bus"]
            # line_impedance = graph.get_edge_data(start_bus, next_bus)["r_ohm"]+1j **graph.get_edge_data(
            # start_line_id, next_bus)["x_ohm"]
        else:
            next_bus = self.net.line.at[start_line_id, "from_bus"]
            # line_impedance = graph.get_edge_data(start_bus, next_bus)["r_ohm"]+1j **graph.get_edge_data(start_line_id, next_bus)["x_ohm"]

        line_value = list(graph.get_edge_data(start_bus, next_bus).values())[0]
        first_line_impedance = line_value["r_ohm"] + 1j * line_value["x_ohm"]
        first_zone_line_impedance = first_line_impedance * 0.9
        associated_lines = [first_zone_line_impedance]

        # Variables to track impedance at different depths
        second_line_impedance = 0
        second_zone_line_impedance = None
        third_zone_line_impedance = None
        second_depth_buses = []
        current_depth_index = 1

        # Traverse from the second bus and find subsequent lines
        previous_bus = start_bus
        current_bus = next_bus

        second_step_reach_parallel_flag = False

        while current_depth_index < depth:
            min_weight = float("inf")
            current_depth_index += 1

            if current_depth_index == 2:

                depth_2_edges = graph.edges(current_bus, data=True)
                valid_forward_edge_found = (
                    False  # Track if any non-backward edge is found
                )

                for from_bus, to_bus, data in depth_2_edges:
                    if to_bus == previous_bus:
                        continue  # Avoid going backward

                    valid_forward_edge_found = True

                    # Store all buses and impedances for depth 2
                    second_depth_buses.append(to_bus)

                    # Check if there are parallel lines between from_bus and to_bus
                    parallel_line_flag = (
                        len(
                            [
                                e
                                for e in depth_2_edges
                                if (e[0] == from_bus and e[1] == to_bus)
                                or (e[0] == to_bus and e[1] == from_bus)
                            ]
                        )
                        > 1
                    )

                    if parallel_line_flag:
                        if data["weight"] * 0.5 < min_weight:
                            second_step_reach_parallel_flag = True
                            min_weight = data["weight"] * 0.5
                            second_line_impedance = data["r_ohm"] + 1j * data["x_ohm"]
                            second_zone_line_impedance = 0.9 * (
                                first_line_impedance + second_line_impedance * 0.5
                            )
                            third_zone_line_impedance = 1.1 * (
                                first_line_impedance + second_line_impedance
                            )
                            # next_bus = to_bus
                    else:
                        if data["weight"] < min_weight:
                            min_weight = data["weight"]
                            second_line_impedance = data["r_ohm"] + 1j * data["x_ohm"]
                            second_zone_line_impedance = 0.9 * (
                                first_line_impedance + second_line_impedance * 0.9
                            )
                            # next_bus = to_bus
                # If no valid forward edges were found, set defaults for second and third zone impedances
                if not valid_forward_edge_found:
                    second_zone_line_impedance = first_line_impedance * 1.2
                    third_zone_line_impedance = 0
                    break  # Stop searching further

            elif current_depth_index == 3:
                # double check if the second zone is set and the parallel line case is specially set so no need to set the zone 3 anymore
                if (
                    second_zone_line_impedance is not None
                    and second_step_reach_parallel_flag is not True
                ):
                    for bus_index in second_depth_buses:

                        depth_3_edges = graph.edges(bus_index, data=True)
                        valid_forward_edge_found = (
                            False  # Track if any non-backward edge is found
                        )

                        for from_bus, to_bus, data in depth_3_edges:
                            if to_bus == previous_bus:
                                continue  # Avoid going backward

                            valid_forward_edge_found = True

                            # Check for parallel line again at this depth
                            parallel_line_flag = (
                                len(
                                    [
                                        e
                                        for e in depth_3_edges
                                        if (e[0] == from_bus and e[1] == to_bus)
                                        or (e[0] == to_bus and e[1] == from_bus)
                                    ]
                                )
                                > 1
                            )

                            if parallel_line_flag:
                                if data["weight"] * 0.5 < min_weight:
                                    min_weight = data["weight"] * 0.5
                                    third_zone_line_impedance = 0.9 * (
                                        first_line_impedance
                                        + second_line_impedance * 0.9
                                        + (data["r_ohm"] + 1j * data["x_ohm"])
                                        * 0.9
                                        * 0.9
                                        * 0.5
                                    )
                            else:
                                if data["weight"] < min_weight:
                                    min_weight = data["weight"]
                                    third_zone_line_impedance = 0.9 * (
                                        first_line_impedance
                                        + second_line_impedance * 0.9
                                        + (data["r_ohm"] + 1j * data["x_ohm"])
                                        * 0.9
                                        * 0.9
                                    )
                                    # If no valid forward edges were found, set the default for the third zone
                    if not valid_forward_edge_found:
                        third_zone_line_impedance = 1.1 * (
                            first_line_impedance + second_line_impedance
                        )
                        break
            # else:
            #     print("Please implement a new zone-grading algorithm.")
            previous_bus = current_bus

        associated_lines.append(second_zone_line_impedance)
        associated_lines.append(third_zone_line_impedance)

        return associated_lines

    def check_zone(self, impedance):
        """Determine the protection zone based on impedance"""
        # how to compare two complex numbers depends on our need
        # r_arc = 2.5  # the arc compensation (ohm) value for 110kv for R-setting
        impedance_point = Point(impedance.real, impedance.imag)
        zone1_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.associated_zone_impedance[0].imag
                    * math.tan(math.radians(30)),
                    self.associated_zone_impedance[0].imag,
                ),
                (
                    self.associated_zone_impedance[0].real + self.r_arc,
                    self.associated_zone_impedance[0].imag,
                ),
                (
                    self.associated_zone_impedance[0].real + self.r_arc,
                    -(self.associated_zone_impedance[0].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )
        zone2_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.associated_zone_impedance[1].imag
                    * math.tan(math.radians(30)),
                    self.associated_zone_impedance[1].imag,
                ),
                (
                    self.associated_zone_impedance[1].real + self.r_arc,
                    self.associated_zone_impedance[1].imag,
                ),
                (
                    self.associated_zone_impedance[1].real + self.r_arc,
                    -(self.associated_zone_impedance[1].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )
        zone3_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.associated_zone_impedance[2].imag
                    * math.tan(math.radians(30)),
                    self.associated_zone_impedance[2].imag,
                ),
                (
                    self.associated_zone_impedance[2].real + self.r_arc,
                    self.associated_zone_impedance[2].imag,
                ),
                (
                    self.associated_zone_impedance[2].real + self.r_arc,
                    -(self.associated_zone_impedance[2].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )

        if zone1_polygon.contains(impedance_point) or zone1_polygon.touches(
            impedance_point
        ):
            return "Zone 1"
        elif zone2_polygon.contains(impedance_point) or zone2_polygon.touches(
            impedance_point
        ):
            return "Zone 2"
        elif zone3_polygon.contains(impedance_point) or zone3_polygon.touches(
            impedance_point
        ):
            return "Zone 3"
        return "Out of Zone"

    def check_zone_ref_old(self, impedance):
        """Determine the protection zone based on impedance"""
        # how to compare two complex numbers depends on our need
        # r_arc = 2.5  # the arc compensation (ohm) value for 110kv for R-setting
        impedance_point = Point(impedance.real, impedance.imag)
        zone1_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.reference_zone_impedance[0].imag * math.tan(math.radians(30)),
                    self.reference_zone_impedance[0].imag,
                ),
                (
                    self.reference_zone_impedance[0].real + self.r_arc,
                    self.reference_zone_impedance[0].imag,
                ),
                (
                    self.reference_zone_impedance[0].real + self.r_arc,
                    -(self.reference_zone_impedance[0].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )
        zone2_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.reference_zone_impedance[1].imag * math.tan(math.radians(30)),
                    self.reference_zone_impedance[1].imag,
                ),
                (
                    self.reference_zone_impedance[1].real + self.r_arc,
                    self.reference_zone_impedance[1].imag,
                ),
                (
                    self.reference_zone_impedance[1].real + self.r_arc,
                    -(self.reference_zone_impedance[1].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )
        zone3_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.reference_zone_impedance[2].imag * math.tan(math.radians(30)),
                    self.reference_zone_impedance[2].imag,
                ),
                (
                    self.reference_zone_impedance[2].real + self.r_arc,
                    self.reference_zone_impedance[2].imag,
                ),
                (
                    self.reference_zone_impedance[2].real + self.r_arc,
                    -(self.reference_zone_impedance[2].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )

        if zone1_polygon.contains(impedance_point) or zone1_polygon.touches(
            impedance_point
        ):
            return "Zone 1"
        elif zone2_polygon.contains(impedance_point) or zone2_polygon.touches(
            impedance_point
        ):
            return "Zone 2"
        elif zone3_polygon.contains(impedance_point) or zone3_polygon.touches(
            impedance_point
        ):
            return "Zone 3"
        return "Out of Zone"

    def check_zone_ref(self, impedance):
        """Determine the protection zone based on impedance, skipping invalid zones with None, 0, or 0j impedance."""

        impedance_point = Point(impedance.real, impedance.imag)
        result_is_valid = (
            True  # Flag indicating if the result is based on a valid zone check
        )

        # Helper function to check if a reference zone impedance is valid
        def is_valid_impedance(zone_impedance):
            return (
                zone_impedance is not None
                and zone_impedance != 0
                and zone_impedance != 0j
            )

        # Initialize polygons for zones if they have valid impedance references
        zone1_polygon = zone2_polygon = zone3_polygon = None

        if is_valid_impedance(self.reference_zone_impedance[0]):
            zone1_polygon = Polygon(
                [
                    (0, 0),
                    (
                        -self.reference_zone_impedance[0].imag
                        * math.tan(math.radians(30)),
                        self.reference_zone_impedance[0].imag,
                    ),
                    (
                        self.reference_zone_impedance[0].real + self.r_arc,
                        self.reference_zone_impedance[0].imag,
                    ),
                    (
                        self.reference_zone_impedance[0].real + self.r_arc,
                        -(self.reference_zone_impedance[0].real + self.r_arc)
                        * math.tan(math.radians(22)),
                    ),
                ]
            )

        if is_valid_impedance(self.reference_zone_impedance[1]):
            zone2_polygon = Polygon(
                [
                    (0, 0),
                    (
                        -self.reference_zone_impedance[1].imag
                        * math.tan(math.radians(30)),
                        self.reference_zone_impedance[1].imag,
                    ),
                    (
                        self.reference_zone_impedance[1].real + self.r_arc,
                        self.reference_zone_impedance[1].imag,
                    ),
                    (
                        self.reference_zone_impedance[1].real + self.r_arc,
                        -(self.reference_zone_impedance[1].real + self.r_arc)
                        * math.tan(math.radians(22)),
                    ),
                ]
            )

        if is_valid_impedance(self.reference_zone_impedance[2]):
            zone3_polygon = Polygon(
                [
                    (0, 0),
                    (
                        -self.reference_zone_impedance[2].imag
                        * math.tan(math.radians(30)),
                        self.reference_zone_impedance[2].imag,
                    ),
                    (
                        self.reference_zone_impedance[2].real + self.r_arc,
                        self.reference_zone_impedance[2].imag,
                    ),
                    (
                        self.reference_zone_impedance[2].real + self.r_arc,
                        -(self.reference_zone_impedance[2].real + self.r_arc)
                        * math.tan(math.radians(22)),
                    ),
                ]
            )

        # Check if the impedance point is within any of the defined polygons
        if zone1_polygon and (
            zone1_polygon.contains(impedance_point)
            or zone1_polygon.touches(impedance_point)
        ):
            return "Zone 1", result_is_valid
        elif zone2_polygon and (
            zone2_polygon.contains(impedance_point)
            or zone2_polygon.touches(impedance_point)
        ):
            return "Zone 2", result_is_valid
        elif zone3_polygon and (
            zone3_polygon.contains(impedance_point)
            or zone3_polygon.touches(impedance_point)
        ):
            return "Zone 3", result_is_valid
        elif zone1_polygon and zone2_polygon and zone3_polygon:
            return "Out of Zone", result_is_valid

        # If the corresponding zone has no valid data, it returns the flag indicating the result is not based on a valid zone check
        return None, False

    def check_zone_with_mag_angle(self, magnitude, angle):
        """Determine the protection zone based on impedance"""
        # how to compare two complex numbers depends on our need
        # r_arc = 2.5  # the arc compensation (ohm) value for 110kv for R-setting
        impedance_point = Point(
            magnitude * math.cos(angle * math.pi / 180),
            magnitude * math.sin(angle * math.pi / 180),
        )
        # try to not let the result precision affect the judgement
        impedance_point = Point(
            np.floor(impedance_point.x * 100) / 100,
            np.floor(impedance_point.y * 100) / 100,
        )
        zone1_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.associated_zone_impedance[0].imag
                    * math.tan(math.radians(30)),
                    self.associated_zone_impedance[0].imag,
                ),
                (
                    self.associated_zone_impedance[0].real + self.r_arc,
                    self.associated_zone_impedance[0].imag,
                ),
                (
                    self.associated_zone_impedance[0].real + self.r_arc,
                    -(self.associated_zone_impedance[0].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )
        zone2_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.associated_zone_impedance[1].imag
                    * math.tan(math.radians(30)),
                    self.associated_zone_impedance[1].imag,
                ),
                (
                    self.associated_zone_impedance[1].real + self.r_arc,
                    self.associated_zone_impedance[1].imag,
                ),
                (
                    self.associated_zone_impedance[1].real + self.r_arc,
                    -(self.associated_zone_impedance[1].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )
        zone3_polygon = Polygon(
            [
                (0, 0),
                (
                    -self.associated_zone_impedance[2].imag
                    * math.tan(math.radians(30)),
                    self.associated_zone_impedance[2].imag,
                ),
                (
                    self.associated_zone_impedance[2].real + self.r_arc,
                    self.associated_zone_impedance[2].imag,
                ),
                (
                    self.associated_zone_impedance[2].real + self.r_arc,
                    -(self.associated_zone_impedance[2].real + self.r_arc)
                    * math.tan(math.radians(22)),
                ),
            ]
        )

        if zone1_polygon.contains(impedance_point) or zone1_polygon.touches(
            impedance_point
        ):
            return "Zone 1"
        elif zone2_polygon.contains(impedance_point) or zone2_polygon.touches(
            impedance_point
        ):
            return "Zone 2"
        elif zone3_polygon.contains(impedance_point) or zone3_polygon.touches(
            impedance_point
        ):
            return "Zone 3"
        return "Out of Zone"


def setup_protection_zones(net, excel_file):
    # initialize the protection_device parameters and return the protection_device lists
    """for calculating the zones, it is necessary to not consider the intermediate nodes"""
    for line in net.line.itertuples():
        if (
            net.bus.loc[line.from_bus, "type"] == "n"
            or net.bus.loc[line.to_bus, "type"] == "n"
        ):
            # print(line.Index)
            net.line.at[line.Index, "in_service"] = False
    """ line between bus c and d has a few virtual buses in between which is not consider in the zone settings"""
    line_data = pd.read_excel(excel_file, sheet_name="line_data", index_col=0)
    line_between_C_D = pp.create_line_from_parameters(
        net,
        from_bus=2,
        to_bus=3,
        length_km=line_data.at[6, "length_km"]
        + line_data.at[7, "length_km"]
        + line_data.at[8, "length_km"]
        + line_data.at[9, "length_km"]
        + line_data.at[10, "length_km"]
        + line_data.at[11, "length_km"],
        r_ohm_per_km=line_data.at[6, "r_ohm_per_km"],
        x_ohm_per_km=line_data.at[6, "x_ohm_per_km"],
        c_nf_per_km=line_data.at[6, "c_nf_per_km"],
        r0_ohm_per_km=line_data.at[6, "r0_ohm_per_km"],
        x0_ohm_per_km=line_data.at[6, "x0_ohm_per_km"],
        c0_nf_per_km=line_data.at[6, "c0_nf_per_km"],
        max_i_ka=line_data.at[6, "max_i_ka"],
        parallel=line_data.at[6, "parallel"],
    )

    # distance_protection_data = pd.read_excel(excel_file, sheet_name='dist_protect_data _simple', index_col=0)
    distance_protection_data = pd.read_excel(
        excel_file, sheet_name="dist_protect_data_complex", index_col=0
    )
    Protection_devices = {}
    for idx in distance_protection_data.index:
        if (
            net.line.at[distance_protection_data.at[idx, "first_line_id"], "in_service"]
            == False
        ):
            continue
        Protection_devices[idx] = ProtectionDevice(
            device_id=distance_protection_data.at[idx, "device_id"],
            bus_id=distance_protection_data.at[idx, "bus_id"],
            first_line_id=distance_protection_data.at[idx, "first_line_id"],
            replaced_line_id=distance_protection_data.at[idx, "replaced_line_id"],
            net=net,
        )

    """ after initialing the protection device parameter, the line between c and d is not needed anymore"""
    # recover the line
    net.line.drop(line_between_C_D, inplace=True)
    for line in net.line.itertuples():
        if (
            net.bus.loc[line.from_bus, "type"] == "n"
            or net.bus.loc[line.to_bus, "type"] == "n"
        ):
            # print(line.Index)
            net.line.at[line.Index, "in_service"] = True

    # keep the associated_zone_impedance but replace the associated lines
    for device in Protection_devices.values():
        device.update_associated_line()

    return Protection_devices


""" Function defined for the fault simulation: start """


def find_affected_devices(line_id, protection_devices, net):
    """
    Find protection devices associated with a given line and its adjacent lines
    (up to two levels of adjacency).

    Parameters:
    - line_id: The ID of the center line for which we are finding affected devices.
    - protection_devices: DataFrame of protection devices with an "associated_line_id" column.
    - net: The network data structure containing line and bus information.

    Returns:
    - List of affected protection devices.
    """
    # Find the buses connected to the center line
    center_from_bus = net.line.at[line_id, "from_bus"]
    center_to_bus = net.line.at[line_id, "to_bus"]

    # Step 1: Find adjacent lines (direct connections)
    adjacent_lines = set(
        net.line[
            (
                net.line["from_bus"].isin([center_from_bus, center_to_bus])
                | net.line["to_bus"].isin([center_from_bus, center_to_bus])
            )
            & (net.line.index != line_id)  # Exclude the center line itself
        ].index
    )

    # Step 2: Find lines adjacent to the adjacent lines (second level of adjacency)
    second_level_buses = set(
        net.line.loc[
            list(adjacent_lines), ["from_bus", "to_bus"]
        ].values.flatten()  # Convert to list
    )
    second_level_lines = set(
        net.line[
            (
                net.line["from_bus"].isin(second_level_buses)
                | net.line["to_bus"].isin(second_level_buses)
            )
            & ~net.line.index.isin(
                adjacent_lines | {line_id}
            )  # Exclude already considered lines
        ].index
    )

    # Combine all relevant line IDs
    relevant_lines = {line_id} | adjacent_lines | second_level_lines

    # Step 3: Filter and return a dictionary of protection devices
    affected_devices = {
        device_id: device
        for device_id, device in protection_devices.items()
        if device.associated_line_id in relevant_lines
    }

    return affected_devices


def filter_and_save_devices_by_line_id(devices, line_id):
    """Filters the list of ProtectionDevice objects based on the associated_line_id and saves the associated_line_id."""
    matching_devices = []
    saved_associated_line_ids = []

    for device in devices.values():
        if device.associated_line_id == line_id:
            matching_devices.append(device)
            saved_associated_line_ids.append(device.associated_line_id)

    return matching_devices, saved_associated_line_ids


def find_line_id_between_buses(bus_id, temp_bus, net):
    # Find the line_id that connects bus_id and temp_bus
    for idx, line in net.line.iterrows():
        if (line["from_bus"] == bus_id and line["to_bus"] == temp_bus) or (
            line["from_bus"] == temp_bus and line["to_bus"] == bus_id
        ):
            return idx
    return None  # Return None or raise an error if no such line is found


def temporally_update_associated_line_id(matching_devices, temp_bus, net):
    if matching_devices is not None:
        for device in matching_devices:
            new_line_id = find_line_id_between_buses(device.bus_id, temp_bus, net)
            # update the device's associated line id
            device.associated_line_id = new_line_id


def recover_associated_line_id(matching_devices, saved_associated_line_ids):
    if matching_devices is not None:
        idx = 0
        for device in matching_devices:
            # update the device's associated line id
            device.associated_line_id = saved_associated_line_ids[idx]
            idx += 1


def convert_to_directed(
    g_undirected, initial_direction, fault_bus, fault_line_on_doubleline_flag
):
    g_directed = nx.DiGraph()  # Initialize a directed graph
    keys_to_extract = ["weight", "r_ohm", "x_ohm"]
    # Copy all other edges from the undirected graph, preserving their attributes
    for u, v in g_undirected.edges():
        attrs_temp = g_undirected.get_edge_data(u, v)
        required_attrs_temp = {
            key: next(iter(attrs_temp.values()))[key] for key in keys_to_extract
        }
        g_directed.add_edge(u, v, **required_attrs_temp)
        g_directed.add_edge(v, u, **required_attrs_temp)

    for neighbor in g_undirected.neighbors(initial_direction[0]):
        if neighbor == initial_direction[1]:
            if g_directed.has_edge(initial_direction[1], initial_direction[0]):
                g_directed.remove_edge(initial_direction[1], initial_direction[0])
        else:
            if g_directed.has_edge(initial_direction[0], neighbor):
                g_directed.remove_edge(initial_direction[0], neighbor)
    if fault_line_on_doubleline_flag:
        if initial_direction == (0, 1):
            if g_directed.has_edge(1, fault_bus):
                g_directed.remove_edge(1, fault_bus)
        elif initial_direction == (1, 0):
            if g_directed.has_edge(0, fault_bus):
                g_directed.remove_edge(0, fault_bus)

    return g_directed


def calculate_impedance(net, device, from_bus, to_bus, fault_line_on_doubleline_flag):
    """Calculate the impedance between two buses. Shortest distance"""
    graph = top.create_nxgraph(
        net, include_lines=True, include_impedances=True, calc_branch_impedances=True
    )
    initial_associated_line = net.line.loc[device.associated_line_id]
    initial_from_bus = from_bus
    initial_to_bus = (
        initial_associated_line.from_bus
        if initial_from_bus != initial_associated_line.from_bus
        else initial_associated_line.to_bus
    )

    # Set initial direction based on device's associated line
    initial_direction = (initial_from_bus, initial_to_bus)
    directed_graph = convert_to_directed(
        graph, initial_direction, to_bus, fault_line_on_doubleline_flag
    )

    total_impedance = 0
    try:
        # Ensure the path exists and retrieve the shortest path based on the impedance
        path = nx.dijkstra_path(
            directed_graph, source=from_bus, target=to_bus, weight="weight"
        )
        # start to consider several complicated cases related with parallel lines
        # case 1 bypassing the external grid
        # Get bus numbers connected to the external grid
        external_grid_buses = net.ext_grid["bus"].tolist()

        # Check if the path bypasses any external grid bus
        for bus in external_grid_buses:
            if (
                bus in path
                and path.index(bus) != 0
                and path.index(bus) != len(path) - 1
            ):
                # print(
                #     f"Path bypasses one of the external grid buses. Device {device.device_id} should not be considered.")
                return None
        path_bus_pairs = set(tuple(sorted([u, v])) for u, v in zip(path[:-1], path[1:]))
        # Identify all bus pairs with parallel lines
        in_service_lines = net.line[net.line["in_service"]]
        parallel_lines = in_service_lines[
            in_service_lines.duplicated(subset=["from_bus", "to_bus"], keep=False)
        ]
        parallel_bus_pairs = set(
            tuple(sorted([row["from_bus"], row["to_bus"]]))
            for _, row in parallel_lines.iterrows()
        )
        # right now the 0 and 1 is hard coded, it will be fixed later
        path_involves_segments_other_than_0_to_bus = (
            len(path_bus_pairs.difference({tuple(sorted([0, to_bus]))})) > 0
        )
        path_involves_segments_other_than_1_to_bus = (
            len(path_bus_pairs.difference({tuple(sorted([1, to_bus]))})) > 0
        )
        parallel_in_path = path_bus_pairs.intersection(parallel_bus_pairs)
        for u, v in zip(path[:-1], path[1:]):
            bus_pair = tuple(sorted([u, v]))
            # Case 2 if parallel line is in path
            if parallel_in_path is not None and bus_pair in parallel_in_path:

                # Calculate combined impedance of parallel lines
                combined_impedance_r = directed_graph.get_edge_data(u, v)["r_ohm"] * 0.5
                combined_impedance_x = directed_graph.get_edge_data(u, v)["x_ohm"] * 0.5
                total_impedance += combined_impedance_r + 1j * combined_impedance_x
            # Case 3 Fault on Parallel Line
            elif (
                fault_line_on_doubleline_flag
                and tuple(sorted([0, to_bus])) == bus_pair
                and path_involves_segments_other_than_0_to_bus
            ):
                line_part1_value = list(graph.get_edge_data(0, to_bus).values())[0]
                line_part2_value = list(graph.get_edge_data(1, to_bus).values())[0]
                line_part3_value = list(graph.get_edge_data(0, 1).values())[0]
                combined_impedance_r = 1 / (
                    1 / (line_part1_value["r_ohm"] + 1e-12)
                    + 1
                    / (line_part2_value["r_ohm"] + line_part3_value["r_ohm"] + 1e-12)
                )
                combined_impedance_x = 1 / (
                    1 / (line_part1_value["x_ohm"] + 1e-12)
                    + 1
                    / (line_part2_value["x_ohm"] + line_part3_value["x_ohm"] + 1e-12)
                )
                total_impedance += combined_impedance_r + 1j * combined_impedance_x
            elif (
                fault_line_on_doubleline_flag
                and tuple(sorted([1, to_bus])) == bus_pair
                and path_involves_segments_other_than_1_to_bus
            ):
                line_part1_value = list(graph.get_edge_data(1, to_bus).values())[0]
                line_part2_value = list(graph.get_edge_data(0, to_bus).values())[0]
                line_part3_value = list(graph.get_edge_data(0, 1).values())[0]
                combined_impedance_r = 1 / (
                    1 / (line_part1_value["r_ohm"] + 1e-12)
                    + 1
                    / (line_part2_value["r_ohm"] + line_part3_value["r_ohm"] + 1e-12)
                )
                combined_impedance_x = 1 / (
                    1 / (line_part1_value["x_ohm"] + 1e-12)
                    + 1
                    / (line_part2_value["x_ohm"] + line_part3_value["x_ohm"] + 1e-12)
                )
                total_impedance += combined_impedance_r + 1j * combined_impedance_x
            else:
                line_value = directed_graph.get_edge_data(u, v)
                total_impedance += line_value["r_ohm"] + 1j * line_value["x_ohm"]

    except nx.NetworkXNoPath:
        # print(f"No path available from bus {from_bus} to bus {to_bus}.")
        total_impedance = None  # Indicate that no path exists.
    return total_impedance


""" Function defined for the fault simulation: end """


# # this is all results export
def simulate_faults_along_line(
    net,
    line_id,
    affected_devices,
    fault_line_on_doubleline_flag,
    interval_km=0.25,
    plot_powerflow_flag=False,
):
    """Simulate faults along a line by adding temporary buses at specified intervals."""
    device_data_dict = []
    # Make original line out of service
    line = net.line.loc[line_id]
    net.line.at[line_id, "in_service"] = False
    # evenly distribute the faults in the line
    line_length = line.length_km
    num_faults = int(line_length / interval_km) - 1
    # temporally change some parameters of the protection devices due to the fault. for simple calculation
    matching_devices, saved_ids = filter_and_save_devices_by_line_id(
        affected_devices, line_id
    )
    # if the fault bus needs to be plotted, the geodata is neccessary unless it will raise index Error
    start_geo_x, start_geo_y = (
        net.bus_geodata.at[net.line.at[line_id, "from_bus"], "x"],
        net.bus_geodata.at[net.line.at[line_id, "from_bus"], "y"],
    )
    end_geo_x, end_geo_y = (
        net.bus_geodata.at[net.line.at[line_id, "to_bus"], "x"],
        net.bus_geodata.at[net.line.at[line_id, "to_bus"], "y"],
    )
    # if the sensed results are different from the calculated results, it is recorded
    # sense_wrong_dict = {}
    if num_faults > 0:

        fault_locations = [interval_km * i for i in range(1, num_faults)] or [
            line_length / 2
        ]

        for fault_location in fault_locations:
            # Create a temporary bus at fault location
            temp_bus = pp.create_bus(
                net,
                vn_kv=HV,
                type="n",
                name="fault_bus",
                geodata=(
                    start_geo_x
                    + (end_geo_x - start_geo_x) * float(fault_location) / line_length,
                    start_geo_y
                    + (end_geo_y - start_geo_y) * float(fault_location) / line_length,
                ),
            )
            # Split the line at the fault location,copy all the other parameters of the original line to the new line
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
                length_km=line_length - fault_location,
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
            # after adding the fault bus into the network, simulate a three-phase short circuit at the temporary bus
            sc.calc_sc(
                net,
                fault="3ph",
                bus=temp_bus,
                branch_results=True,
                return_all_currents=True,
            )
            if plot_powerflow_flag:
                plot_short_circuit_results(net, line_id)
            # change the parameters of the protection device
            temporally_update_associated_line_id(matching_devices, temp_bus, net)
            for device in affected_devices:
                # according to the line length to calcualte the distance the protection devices supposed to sense
                impedance = calculate_impedance(
                    net,
                    affected_devices[device],
                    affected_devices[device].bus_id,
                    temp_bus,
                    fault_line_on_doubleline_flag,
                )
                if impedance is None:
                    # print(
                    #     f"Impedance calculation returned None for device {affected_devices[device].device_id} at bus {affected_devices[device].bus_id}. Skipping this fault scenario.")
                    continue  # Skip to the next fault scenario
                zone_calculated, valid_result_flag = affected_devices[
                    device
                ].check_zone_ref(impedance)
                # this flag is free from the misjudgement from 0 or None vaule setting in the protection zone
                if valid_result_flag == False:
                    continue
                # Get the impedance at the protection device through the line results
                if (
                    affected_devices[device].bus_id
                    == net.line.loc[affected_devices[device].associated_line_id][
                        "from_bus"
                    ]
                ):
                    vm_pu = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["vm_from_pu"].item()
                    ikss_ka = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["ikss_from_ka"].item()
                    va_degree = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["va_from_degree"].item()
                    ikss_degree = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["ikss_from_degree"].item()
                elif (
                    affected_devices[device].bus_id
                    == net.line.loc[affected_devices[device].associated_line_id][
                        "to_bus"
                    ]
                ):
                    vm_pu = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["vm_to_pu"].item()
                    ikss_ka = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["ikss_to_ka"].item()
                    va_degree = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["va_to_degree"].item()
                    ikss_degree = net.res_line_sc.loc[
                        affected_devices[device].associated_line_id
                    ]["ikss_to_degree"].item()
                else:
                    print(
                        f"the line {affected_devices[device].associated_line_id} result is not existed"
                    )
                    return None, None, None
                # calculate the magnitude and the angle of the impedance
                r_sensed = vm_pu * HV / ((ikss_ka + 1e-9) * 3**0.5)
                angle_sensed = va_degree - ikss_degree
                zone_sensed = affected_devices[device].check_zone_with_mag_angle(
                    r_sensed, angle_sensed
                )

                # try to not record the result that is not in the forward direction
                if angle_sensed < 0:
                    continue

                device_data = {
                    "Device ID": affected_devices[device].device_id,
                    "Fault_line_id": line_id,
                    "Referenced_bus": line.from_bus,
                    "Distance_from_bus": fault_location,
                    "Impedance_calculated": impedance,
                    "zone_calculated": zone_calculated,
                    "r_sensed": r_sensed,
                    "angle_sensed": angle_sensed,
                    "zone_sensed": zone_sensed,
                    "same_zone_detection": zone_calculated == zone_sensed,
                }
                # Append individual device data to the main list (flattened)
                device_data_dict.append(device_data)
            recover_associated_line_id(matching_devices, saved_ids)
            # Remove temporary buses and associated lines after analysis.
            net.line.drop(temp_line_part1, inplace=True)
            net.line.drop(temp_line_part2, inplace=True)
            net.bus.drop(temp_bus, inplace=True)

    return device_data_dict


def simulate_faults_for_all_lines(net, protection_devices, interval_km=0.25):
    """Simulate faults along all in-service lines and collect data for protection devices."""
    protection_data = []
    # Identify double lines by finding bus pairs with multiple lines in service
    double_line_pairs = (
        net.line[net.line["in_service"]]
        .groupby(["from_bus", "to_bus"])
        .filter(lambda x: len(x) > 1)
    )
    double_line_ids = set(double_line_pairs.index)

    for line_id in net.line.index:
        if net.line.at[line_id, "in_service"]:  # Only consider in-service lines
            print(f"Simulating faults along line {line_id}")

            # Assuming that affected devices are consistent along the line
            affected_devices = find_affected_devices(line_id, protection_devices, net)

            # Set the flag if the current line_id is part of any double-line pair
            fault_line_on_doubleline_flag = line_id in double_line_ids

            # Simulate faults along the line
            device_data = simulate_faults_along_line(
                net,
                line_id,
                affected_devices,
                fault_line_on_doubleline_flag,
                interval_km,
            )

            # Restore the original line to service after analysis
            net.line.at[line_id, "in_service"] = True

            protection_data.extend(device_data)

    return protection_data


"""start: plot the short circuit simulation power flow"""


def create_voltage_dict(net):
    # Initialize an empty dictionary to store bus voltages
    bus_voltage_dict = {}

    # Iterate through the lines in res_line_sc
    for idx in net.res_line_sc.index:
        line_id = idx
        if isinstance(idx, tuple):
            line_id = idx[0]
        # Get bus IDs for the line from the net.line DataFrame using idx
        if net.line.at[line_id, "in_service"]:
            from_bus = net.line.loc[line_id, "from_bus"]
            to_bus = net.line.loc[line_id, "to_bus"]

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
        line_id = idx
        if isinstance(idx, tuple):
            line_id = idx[0]
        if net.line.at[line_id, "in_service"]:
            current_and_direction_dict = {}
            # Get the current magnitude
            ikss = row["ikss_ka"]
            current_and_direction_dict["current"] = ikss

            # Get the direction of the power flow
            from_bus = net.line.loc[line_id, "from_bus"]
            to_bus = net.line.loc[line_id, "to_bus"]

            # Get voltage angles
            voltage_angle_from = net.res_line_sc.loc[idx, "va_from_degree"]

            # get current angle
            # current_angle_to = net.res_line_sc.loc[idx, "ikss_to_degree"]
            current_angle_from = net.res_line_sc.loc[idx, "ikss_from_degree"]

            direction = (from_bus, to_bus)
            # Determine the direction of current flow
            if math.cos((voltage_angle_from - current_angle_from) * math.pi / 180) > 0:
                direction = (from_bus, to_bus)  # Current flows from from_bus to to_bus
            else:
                direction = (to_bus, from_bus)  # Current flows from to_bus to from_bus

            # Create dictionary entries for both buses
            current_and_direction_dict["direction"] = direction
            current_dict[line_id] = current_and_direction_dict

    return current_dict


def my_simple_plot(
    net,
    respect_switches=False,
    line_width=1.0,
    bus_size=1.0,
    ext_grid_size=1.0,
    trafo_size=1.0,
    plot_loads=False,
    plot_gens=False,
    plot_sgens=False,
    sgn_oritation=0,
    load_size=1.0,
    gen_size=1.0,
    sgen_size=1.0,
    switch_size=2.0,
    switch_distance=1.0,
    plot_line_switches=False,
    scale_size=True,
    bus_color="b",
    line_color="grey",
    dcline_color="c",
    trafo_color="k",
    ext_grid_color="y",
    switch_color="k",
    library="igraph",
    show_plot=True,
    ax=None,
):
    """
    inherit from the simple plot function but add the component direction input parameter
    """
    # don't hide lines if switches are plotted
    if plot_line_switches:
        respect_switches = False

    # create geocoord if none are available
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning(
            "No or insufficient geodata available --> Creating artificial coordinates."
            + " This may take some time"
        )
        pplot.create_generic_coordinates(
            net, respect_switches=respect_switches, library=library
        )

    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        sizes = pplot.get_collection_sizes(
            net,
            bus_size,
            ext_grid_size,
            trafo_size,
            load_size,
            sgen_size,
            switch_size,
            switch_distance,
            gen_size,
        )
        bus_size = sizes["bus"]
        ext_grid_size = sizes["ext_grid"]
        trafo_size = sizes["trafo"]
        sgen_size = sizes["sgen"]
        load_size = sizes["load"]
        switch_size = sizes["switch"]
        switch_distance = sizes["switch_distance"]
        gen_size = sizes["gen"]

    # create bus collections to plot
    bc = create_bus_collection(net, size=bus_size, color=bus_color, zorder=10)

    # if bus geodata is available, but no line geodata
    use_bus_geodata = len(net.line_geodata) == 0
    in_service_lines = net.line[net.line.in_service].index
    nogolines = (
        set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)])
        if respect_switches
        else set()
    )
    plot_lines = in_service_lines.difference(nogolines)
    plot_dclines = net.dcline.in_service

    # create line collections
    lc = create_line_collection(
        net,
        plot_lines,
        color=line_color,
        linewidths=line_width,
        use_bus_geodata=use_bus_geodata,
    )
    collections = [bc, lc]

    # create dcline collections
    if len(net.dcline) > 0:
        dclc = create_dcline_collection(
            net, plot_dclines, color=dcline_color, linewidths=line_width
        )
        collections.append(dclc)

    # create ext_grid collections
    # eg_buses_with_geo_coordinates = set(net.ext_grid.bus.values) & set(net.bus_geodata.index)
    if len(net.ext_grid) > 0:
        sc = create_ext_grid_collection(
            net,
            size=ext_grid_size,
            orientation=0,
            ext_grids=net.ext_grid.index,
            patch_edgecolor=ext_grid_color,
            zorder=11,
        )
        collections.append(sc)

    # create trafo collection if trafo is available
    trafo_buses_with_geo_coordinates = [
        t
        for t, trafo in net.trafo.iterrows()
        if trafo.hv_bus in net.bus_geodata.index
        and trafo.lv_bus in net.bus_geodata.index
    ]
    if len(trafo_buses_with_geo_coordinates) > 0:
        tc = create_trafo_collection(
            net, trafo_buses_with_geo_coordinates, color=trafo_color, size=trafo_size
        )
        collections.append(tc)

    # create trafo3w collection if trafo3w is available
    trafo3w_buses_with_geo_coordinates = [
        t
        for t, trafo3w in net.trafo3w.iterrows()
        if trafo3w.hv_bus in net.bus_geodata.index
        and trafo3w.mv_bus in net.bus_geodata.index
        and trafo3w.lv_bus in net.bus_geodata.index
    ]
    if len(trafo3w_buses_with_geo_coordinates) > 0:
        tc = create_trafo3w_collection(
            net, trafo3w_buses_with_geo_coordinates, color=trafo_color
        )
        collections.append(tc)

    if plot_line_switches and len(net.switch):
        sc = create_line_switch_collection(
            net,
            size=switch_size,
            distance_to_bus=switch_distance,
            use_line_geodata=not use_bus_geodata,
            zorder=12,
            color=switch_color,
        )
        collections.append(sc)

    if plot_sgens and len(net.sgen):
        in_service_sgens = net.sgen[net.sgen["in_service"]]
        sgc = create_sgen_collection(
            net, size=sgen_size, orientation=sgn_oritation, sgens=in_service_sgens.index
        )
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
            soft_dependency_error(
                str(sys._getframe().f_code.co_name) + "()", "matplotlib"
            )
        plt.show()
    return ax


def plot_short_circuit_results(net, figure_number):
    if net.res_bus_sc.empty or net.res_line_sc.empty:
        raise ValueError(
            "Short circuit results are missing. Please run the short circuit simulation before plotting."
        )
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
    current_values = [info["current"] for info in current_dict.values()]
    min_current = min(current_values)
    max_current = max(current_values)
    current_norm = plt.Normalize(vmin=min_current, vmax=max_current)

    # Base plot with pandapower simple_plot
    ax = my_simple_plot(
        net,
        plot_loads=True,
        scale_size=True,
        load_size=2,
        plot_sgens=True,
        sgen_size=2,
        bus_color="#00000000",
        show_plot=False,
        sgn_oritation=np.pi / 2,
    )

    # Plot bus voltage with darker color means higher voltage
    for bus_id, voltage in voltage_dict.items():
        coords = net.bus_geodata.loc[bus_id]
        color = voltage_cmap(voltage_norm(voltage))
        ax.scatter(
            coords.x,
            coords.y,
            color=color,
            edgecolor="black",
            s=100,
            label=f"Bus {bus_id}: {voltage:.2f} pu",
            zorder=5,
        )

    # Plot current magnitude and direction
    for _, info in current_dict.items():
        direction = info["direction"]
        from_bus, to_bus = direction

        # Get coordinates for the from and to buses
        from_coords = net.bus_geodata.loc[from_bus]
        to_coords = net.bus_geodata.loc[to_bus]
        midpoint_x = (from_coords.x + to_coords.x) / 2
        midpoint_y = (from_coords.y + to_coords.y) / 2
        # Color is difficult to visulize, change it to thickness instead
        current_magnitude = info["current"]
        line_thickness = (current_magnitude / max_current + 1) * 0.04
        ax.text(
            midpoint_x,
            midpoint_y,
            f"{current_magnitude:.2f} kA",
            color="black",
            ha="center",
            va="center",
            fontsize=8,
            fontweight=600,
            zorder=4,
        )

        # Calculate arrow direction and apply a scale factor to make it more readable
        dx, dy = to_coords.x - from_coords.x, to_coords.y - from_coords.y
        length_factor = 0.85  # Adjust length of arrows to improve readability
        dx, dy = dx * length_factor, dy * length_factor

        # Offset for starting point to make the head clearly visible
        offset_factor = 0.05
        offset_x, offset_y = dx * offset_factor, dy * offset_factor

        # Draw the improved arrow with larger head and offset
        ax.arrow(
            from_coords.x + offset_x,
            from_coords.y + offset_y,
            dx,
            dy,
            color="#a36c65",
            width=line_thickness,
            head_width=line_thickness * 4,
            head_length=line_thickness * 4,
            length_includes_head=True,
            zorder=3,
        )

    output_dir = "sc_pf_half_position"
    os.makedirs(output_dir, exist_ok=True)
    # Show plot with legend
    ax.legend()
    # To save figures into the output_dir
    # plt.savefig(os.path.join(output_dir, f'sc_plot_{figure_number}.png'))  # Save as PNG
    plt.show(block=False)


"""end: plot the short circuit simulation power flow"""
