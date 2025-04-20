import pandapower as pp
import numpy as np

# Create a network
net = pp.create_empty_network()
HV = 110  # 110 kV
bus1 = pp.create_bus(net, vn_kv=HV)
bus2 = pp.create_bus(net, vn_kv=HV)
bus3 = pp.create_bus(net, vn_kv=HV)
c_max = 1.1
# Slack bus at 1.0 pu
line1 = pp.create_line(
    net, bus1, bus2, length_km=10, std_type="N2XS(FL)2Y 1x120 RM/35 64/110 kV"
)
line2 = pp.create_line(
    net, bus2, bus3, length_km=20, std_type="N2XS(FL)2Y 1x120 RM/35 64/110 kV"
)
pp.create_ext_grid(
    net,
    bus=bus1,
    vm_pu=1.0,
    va_degree=0,
    s_sc_max_mva=1000,
    rx_max=0.1,  # Required parameters for short-circuit simulation
)
# Add a load to create pre-fault voltage drop
pp.create_load(net, bus3, p_mw=50, q_mvar=20)
# Add a generator to create additional current contribution
pp.create_sgen(
    net,
    bus3,
    p_mw=15,
    q_mvar=5,
    sn_mva=35,
    k=1.2,
    kappa=1.2,
    generator_type="current_source",
)
# Run power flow to get pre-fault voltage at bus2
pp.runpp(net)


import pandapower.shortcircuit as sc
import logging

logging.basicConfig(
    level=logging.ERROR
)  # pandapower original package always brings warning logging
sc.calc_sc(net, bus=bus2, use_pre_fault_voltage=False)
net.res_bus_sc
