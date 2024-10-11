import simbench as sb
import pandas as pd
import numpy as np
import pandapower.timeseries as ts
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData



def load_profiles(net):
    # Define the time steps
    samples_per_month = 31 * 24 * 60 // 15

    """load the load p,q profile from simbench dataset"""
    grid_code = "1-HV-mixed--0-sw"
    profiles = sb.get_absolute_values(sb.get_simbench_net(grid_code), profiles_instead_of_study_cases=True)
    load_p = profiles[("load", "p_mw")]
    load_q = profiles[("load", "q_mvar")]
    load_proportion_1 = 2 / 5
    load_proportion_2 = 2 / 5
    load_proportion_3 = 1 / 5
    # Extract the first month's data
    one_month_data = load_p.iloc[:samples_per_month].sum(axis=1) # Sum the groups with their respective proportions
    load_1_p = one_month_data * load_proportion_1
    load_2_p = one_month_data * load_proportion_2
    load_3_p = one_month_data * load_proportion_3
    load_p_sum = pd.DataFrame({
        'load_1': load_1_p,
        'load_2': load_2_p,
        'load_3': load_3_p
    })

    one_month_data = load_q.iloc[:samples_per_month].sum(axis=1) # Sum the groups with their respective proportions
    load_1_q = one_month_data * load_proportion_1
    load_2_q = one_month_data * load_proportion_2
    load_3_q = one_month_data * load_proportion_3
    load_q_sum= pd.DataFrame({
        'load_1': load_1_q,
        'load_2': load_2_q,
        'load_3': load_3_q
    })

    """create datasource and initialize controller"""
    ds_load = DFData(load_p_sum)
    ConstControl(net, element='load', element_index=net.load.index[0], variable='p_mw', data_source=ds_load,
                profile_name='load_1')
    ConstControl(net, element='load', element_index=net.load.index[1], variable='p_mw', data_source=ds_load,
                profile_name='load_2')
    ConstControl(net, element='load', element_index=net.load.index[2], variable='p_mw', data_source=ds_load,
                profile_name='load_3')
    ds_load = DFData(load_q_sum)
    ConstControl(net, element='load', element_index=net.load.index[0], variable='q_mvar', data_source=ds_load,
                profile_name='load_1')
    ConstControl(net, element='load', element_index=net.load.index[1], variable='q_mvar', data_source=ds_load,
                profile_name='load_2')
    ConstControl(net, element='load', element_index=net.load.index[2], variable='q_mvar', data_source=ds_load,
                profile_name='load_3')

    """add some noise to the wind generation data"""
    df = pd.DataFrame(np.random.normal(1., 0.1, size=(samples_per_month, len(net.sgen.index))),
                index=list(range(samples_per_month)), columns=net.sgen.index) * net.sgen.p_mw.values

    """create datasource from it"""
    ds_sgen = DFData(df)

    """initialising ConstControl controller to update values"""
    ConstControl(net, element='sgen', element_index=net.sgen.index,variable='p_mw', data_source=ds_sgen, profile_name=net.sgen.index)

    """create output writer"""
    ow = ts.OutputWriter(net, output_path="./", output_file_type=".xlsx")
    ow.log_variable("res_load", "p_mw")
    ow.log_variable("res_load", "q_mvar")
    ow.log_variable("res_sgen", "p_mw")
        
    print(net)
    return net