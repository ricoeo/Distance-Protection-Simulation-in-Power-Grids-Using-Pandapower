import simbench as sb
import pandas as pd
import numpy as np
import pandapower.timeseries as ts
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from wetterdienst.provider.dwd.observation import (
    DwdObservationDataset,
    DwdObservationParameter,
    DwdObservationPeriod,
    DwdObservationRequest,
    DwdObservationResolution,
)
from wetterdienst import Parameter, Resolution, Settings
import warnings
from scipy import interpolate


def resample_profiles(profiles, resolution="10min", agg_func="mean"):
    """
    Resamples time series data in the profiles to a specified resolution.

    Parameters:
    - profiles (dict): Dictionary of DataFrames, typically the SimBench profiles.
    - resolution (str): Desired time resolution (e.g., "10min", "1H").
    - agg_func (str): Aggregation function ("mean", "sum", etc.) or callable for resampling.

    Returns:
    - dict: Resampled profiles as a dictionary of DataFrames.
    """
    resampled_profiles = {}

    for key, df in profiles.items():
        # Check if the DataFrame has a time-based index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"The DataFrame for key {key} must have a DatetimeIndex.")

        # Resample the DataFrame
        if agg_func == "mean":
            resampled_df = df.resample(resolution).mean()
        elif agg_func == "sum":
            resampled_df = df.resample(resolution).sum()
        elif callable(agg_func):
            resampled_df = df.resample(resolution).apply(agg_func)
        else:
            raise ValueError("agg_func must be 'mean', 'sum', or a callable function.")

        # Store the resampled DataFrame
        resampled_profiles[key] = resampled_df

    return resampled_profiles


def generate_wind_turbine_power_output(wind_speeds):
    """
    Generate the real and imaginary power outputs for a wind turbine based on input wind speeds.

    This function uses an interpolation of the power coefficient (Ce) at various wind speeds (1 m/s to 25 m/s),
    calculates the real power output for a given wind speed using a simplified wind turbine power formula,
    and then calculates an imaginary power output as half of the real power.

    The formula used for real power output is:
        P_real = 0.5 * air_density * swept_area * wind_speed^3 * Ce / 1e6
    where:
        air_density = 1.225 kg/m³ (density of air at sea level),
        swept_area is the area swept by the turbine rotor blades,
        wind_speed is the input wind speed in m/s,
        Ce is the power coefficient at the given wind speed.

    The imaginary power output is defined as:
        P_imaginary = 0.5 * P_real

    Parameters:
    -----------
    wind_speeds : list or array
        A list or array of wind speeds (in meters per second) for which the power output is calculated.

    Returns:
    --------
    real_power_outputs : list
        A list of real power outputs (in MW) corresponding to the input wind speeds.

    imaginary_power_outputs : list
        A list of imaginary power outputs (in MW) corresponding to the input wind speeds.
    """
    """wind turbine coefficient can be tuned"""
    wind_data = {
        "v_mps": np.arange(1, 26),
        "Ce": [
            0.000,
            0.000,
            0.228,
            0.322,
            0.380,
            0.408,
            0.413,
            0.418,
            0.419,
            0.419,
            0.401,
            0.378,
            0.305,
            0.244,
            0.199,
            0.164,
            0.136,
            0.115,
            0.098,
            0.084,
            0.072,
            0.063,
            0.055,
            0.048,
            0.042,
        ],
    }

    # Create a DataFrame for the coefficients
    wind_df = pd.DataFrame(wind_data)

    # Interpolation function for Ce based on wind speed
    interpolation_function = interpolate.interp1d(
        wind_df["v_mps"], wind_df["Ce"], kind="linear", fill_value="extrapolate"
    )

    air_density = 1.225  # kg/m³ (density of air at sea level)

    # Assuming a rotor diameter of about 100 meters (typical for a 50 MW turbine)
    rotor_diameter = 100  # meters
    swept_area = np.pi * (rotor_diameter / 2) ** 2  # m² (swept area of the rotor)

    # Calculate the real and imaginary power outputs
    real_power_outputs = []
    imaginary_power_outputs = []

    for wind_speed in wind_speeds:
        # Interpolate the Cp (Ce) for the given wind speed
        Cp = interpolation_function(wind_speed)

        # Calculate real power output (simplified)
        real_power_output = (
            0.5 * air_density * swept_area * wind_speed**3 * Cp / 1e6
        )  # In MW

        # Add an imaginary part (half of the real part)
        imaginary_power_output = 0.5 * real_power_output

        real_power_outputs.append(real_power_output)
        imaginary_power_outputs.append(imaginary_power_output)

    return real_power_outputs, imaginary_power_outputs


def load_profiles_old(net):
    # Define the time steps
    samples_per_month = 31 * 24 * 60 // 15

    """load the load p,q profile from simbench dataset"""
    grid_code = "1-HV-mixed--0-sw"
    profiles = sb.get_absolute_values(
        sb.get_simbench_net(grid_code), profiles_instead_of_study_cases=True
    )

    load_p = profiles[("load", "p_mw")]
    load_q = profiles[("load", "q_mvar")]
    load_proportion_1 = 2 / 5
    load_proportion_2 = 2 / 5
    load_proportion_3 = 1 / 5
    # Extract the first month's data
    one_month_data = load_p.iloc[:samples_per_month].sum(
        axis=1
    )  # Sum the groups with their respective proportions
    load_1_p = one_month_data * load_proportion_1
    load_2_p = one_month_data * load_proportion_2
    load_3_p = one_month_data * load_proportion_3
    load_p_sum = pd.DataFrame(
        {"load_1": load_1_p, "load_2": load_2_p, "load_3": load_3_p}
    )

    one_month_data = load_q.iloc[:samples_per_month].sum(
        axis=1
    )  # Sum the groups with their respective proportions
    load_1_q = one_month_data * load_proportion_1
    load_2_q = one_month_data * load_proportion_2
    load_3_q = one_month_data * load_proportion_3
    load_q_sum = pd.DataFrame(
        {"load_1": load_1_q, "load_2": load_2_q, "load_3": load_3_q}
    )

    """create datasource and initialize controller"""
    ds_load = DFData(load_p_sum)
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[0],
        variable="p_mw",
        data_source=ds_load,
        profile_name="load_1",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[1],
        variable="p_mw",
        data_source=ds_load,
        profile_name="load_2",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[2],
        variable="p_mw",
        data_source=ds_load,
        profile_name="load_3",
    )
    ds_load = DFData(load_q_sum)
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[0],
        variable="q_mvar",
        data_source=ds_load,
        profile_name="load_1",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[1],
        variable="q_mvar",
        data_source=ds_load,
        profile_name="load_2",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[2],
        variable="q_mvar",
        data_source=ds_load,
        profile_name="load_3",
    )

    """add some noise to the wind generation data"""
    df = (
        pd.DataFrame(
            np.random.normal(1.0, 0.1, size=(samples_per_month, len(net.sgen.index))),
            index=list(range(samples_per_month)),
            columns=net.sgen.index,
        )
        * net.sgen.p_mw.values
    )

    """create datasource from it"""
    ds_sgen = DFData(df)

    """initialising ConstControl controller to update values"""
    ConstControl(
        net,
        element="sgen",
        element_index=net.sgen.index,
        variable="p_mw",
        data_source=ds_sgen,
        profile_name=net.sgen.index,
    )

    """create output writer"""
    ow = ts.OutputWriter(net, output_path="./", output_file_type=".xlsx")
    ow.log_variable("res_load", "p_mw")
    ow.log_variable("res_load", "q_mvar")
    ow.log_variable("res_sgen", "p_mw")

    print(net)
    return net


# Define start and end dates
start_date = "2023-01-01"
end_date = "2023-02-01"
sample_interval_load = "15min"
sample_interval_wind = "10min"


def load_profiles(net):
    """load the load p,q profile from simbench dataset"""
    grid_code = "1-HV-mixed--0-sw"
    profiles = sb.get_absolute_values(
        sb.get_simbench_net(grid_code), profiles_instead_of_study_cases=True
    )
    # Define the time steps, profiles in the dataset has a 15 min resolution
    sample_interval = "15min"
    num_samples = len(
        pd.date_range(start=start_date, end=end_date, freq=sample_interval)
    )
    load_p = profiles[("load", "p_mw")]
    load_q = profiles[("load", "q_mvar")]
    load_proportion_1 = 2 / 5
    load_proportion_2 = 2 / 5
    load_proportion_3 = 1 / 5

    # Ensure num_samples does not exceed the available data in load_p
    available_samples = len(load_p)

    if num_samples > available_samples:
        warnings.warn(
            f"num_samples ({num_samples}) exceeds the available data length ({available_samples}). Adjusting num_samples to match the data length."
        )
        num_samples = available_samples
    # Extract the first month's data
    one_month_data_origin = load_p.iloc[:num_samples].sum(
        axis=1
    )  # Sum the groups with their respective proportions
    # resample load profile to align with the wind data
    one_month_data = resample_profiles(
        one_month_data_origin, sample_interval_wind, "mean"
    )
    load_1_p = one_month_data * load_proportion_1
    load_2_p = one_month_data * load_proportion_2
    load_3_p = one_month_data * load_proportion_3
    load_p_sum = pd.DataFrame(
        {"load_1": load_1_p, "load_2": load_2_p, "load_3": load_3_p}
    )

    one_month_data_origin = load_q.iloc[:num_samples].sum(
        axis=1
    )  # Sum the groups with their respective proportions
    one_month_data = resample_profiles(
        one_month_data_origin, sample_interval_wind, "mean"
    )
    load_1_q = one_month_data * load_proportion_1
    load_2_q = one_month_data * load_proportion_2
    load_3_q = one_month_data * load_proportion_3
    load_q_sum = pd.DataFrame(
        {"load_1": load_1_q, "load_2": load_2_q, "load_3": load_3_q}
    )

    """create datasource and initialize controller"""
    ds_load = DFData(load_p_sum)
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[0],
        variable="p_mw",
        data_source=ds_load,
        profile_name="load_1",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[1],
        variable="p_mw",
        data_source=ds_load,
        profile_name="load_2",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[2],
        variable="p_mw",
        data_source=ds_load,
        profile_name="load_3",
    )
    ds_load = DFData(load_q_sum)
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[0],
        variable="q_mvar",
        data_source=ds_load,
        profile_name="load_1",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[1],
        variable="q_mvar",
        data_source=ds_load,
        profile_name="load_2",
    )
    ConstControl(
        net,
        element="load",
        element_index=net.load.index[2],
        variable="q_mvar",
        data_source=ds_load,
        profile_name="load_3",
    )

    """get wind data from DWD and extract the exact location wind data"""
    request_hamburg = DwdObservationRequest(
        parameter=Parameter.WIND_SPEED,
        resolution=DwdObservationResolution.MINUTE_10,
        start_date=start_date,
        end_date=end_date,
    )
    # if the method interpolate is used, this line will be useful and please set a reasonable search radius
    # print(request_center_hamburg.interpolate(latlon=(53.70, 9.96)).df.head())
    # Sgn 0
    wind_speed_sgn0 = request_hamburg.summarize(
        latlon=(53.61003560950149, 10.327354606743228)
    ).df

    # Sgn 1
    wind_speed_sgn1 = request_hamburg.summarize(
        latlon=(53.24180742678847, 10.16255969706929)
    ).df

    # Sgn 2
    wind_speed_sgn2 = request_hamburg.summarize(
        latlon=(53.24484512265859, 10.108828220940879)
    ).df

    # Sgn 3
    wind_speed_sgn3 = request_hamburg.summarize(
        latlon=(53.30013422107673, 9.89767081529967)
    ).df

    # Sgn 4
    wind_speed_sgn4 = request_hamburg.summarize(
        latlon=(53.305878809221326, 9.888057779086475)
    ).df

    # Sgn 5
    wind_speed_sgn5 = request_hamburg.summarize(
        latlon=(53.3181860388135, 9.83175285269491)
    ).df

    # Sgn 6
    wind_speed_sgn6 = request_hamburg.summarize(
        latlon=(53.33377010299398, 9.775447926303345)
    ).df

    """add some noise to the wind generation data"""
    df = (
        pd.DataFrame(
            np.random.normal(1.0, 0.1, size=(samples_per_month, len(net.sgen.index))),
            index=list(range(samples_per_month)),
            columns=net.sgen.index,
        )
        * net.sgen.p_mw.values
    )

    """create datasource from it"""
    ds_sgen = DFData(df)

    """initialising ConstControl controller to update values"""
    ConstControl(
        net,
        element="sgen",
        element_index=net.sgen.index,
        variable="p_mw",
        data_source=ds_sgen,
        profile_name=net.sgen.index,
    )

    """create output writer"""
    ow = ts.OutputWriter(net, output_path="./", output_file_type=".xlsx")
    ow.log_variable("res_load", "p_mw")
    ow.log_variable("res_load", "q_mvar")
    ow.log_variable("res_sgen", "p_mw")

    print(net)
    return net
