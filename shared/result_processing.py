import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# Global plot settings
PLOT_CONFIG = {
    "legend_fontsize": 16,  # Font size for the legend
    "axis_labelsize": 18,  # Font size for axis labels
    "tick_labelsize": 13,  # Font size for axis ticks
    "line_width": 2,  # Line width for plots
    "marker_size": 6,  # Marker size for highlighted points
    "fig_size": (12, 6),  # Figure size
    "fig_large_size": (12, 12),
}


def plot_fault_case_ratios(base_path, frac=0.05):
    """
    Reads all results folders inside base_path, extracts fault case ratios, and plots them.

    :param base_path: Path to the 'timeseries_results' directory.
    """
    ratios = pd.DataFrame()

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "custom")

        if not os.path.exists(folder_path):
            break  # Stop when a results_* folder is missing

        excel_path = os.path.join(folder_path, "Value.xlsx")
        if os.path.exists(excel_path):
            # Read the Excel file
            df = pd.read_excel(
                excel_path, usecols=["Total_fault_cases", "Total_cases_analyzed"]
            )

            if (
                "Total_fault_cases" in df.columns
                and "Total_cases_analyzed" in df.columns
            ):
                # Compute the row-wise ratio, avoiding division by zero
                df["Ratio"] = df["Total_fault_cases"] / df["Total_cases_analyzed"]
                df["Ratio"] = df["Ratio"].replace(
                    [float("inf"), -float("inf")], None
                )  # Handle divide-by-zero cases

                # Store all valid ratios from this file
                ratios = pd.concat([ratios, df[["Ratio"]].dropna()], ignore_index=True)
                # valid_ratios = df["Ratio"].dropna().tolist()
                # ratios.extend(valid_ratios)

        i += 1  # Move to the next results_* folder

    if ratios.empty:
        print("NO Data Found")
        return
    # Group data into weekly bins (1008 data points/week)
    WEEKLY_RESOLUTION = 1008
    ratios["Week"] = (ratios.index // WEEKLY_RESOLUTION) + 1

    # Prepare data for boxplot
    grouped = ratios.groupby("Week")["Ratio"].apply(list)
    box_data = grouped.tolist()
    week_labels = [f" {w}" for w in grouped.index]

    # Plot the results
    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    plt.boxplot(box_data, patch_artist=True, widths=0.6)
    # Configure axes
    plt.xticks(
        ticks=range(1, len(week_labels) + 1),
        labels=week_labels,
        rotation=45,
        ha="right",
        fontsize=PLOT_CONFIG["tick_labelsize"],
    )
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])

    plt.xlabel("Week Number", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Fault Case Ratio Distribution", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    # Save the figure as a PDF
    plt.savefig("figure/fault_case_ratios.pdf", format="pdf")
    plt.show()


# Example usage:
# plot_fault_case_ratios("timeseries_results_reference", frac=0.01)


def plot_wind_activepower(base_path):
    # the detail is not kept if the plot is possible to read, but if it is possible to read, the peak the vally detail is lost!
    active_power_df = pd.DataFrame()
    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "res_sgen")

        if not os.path.exists(folder_path):
            break  # Stop when a results_* folder is missing

        excel_path = os.path.join(folder_path, "p_mw.xlsx")
        if os.path.exists(excel_path):
            # Read the Excel file
            df = pd.read_excel(excel_path, index_col=0)
            active_power_df = pd.concat([active_power_df, df], ignore_index=True)
        i += 1
    if active_power_df.empty:
        print("NO Data Found")
        return

    num_points = len(active_power_df)
    xticks = list(range(0, num_points, 4320))  # Mark every 30 days
    xticklabels = [f"{i//144}d" for i in xticks]
    fig, axs = plt.subplots(active_power_df.shape[1], 1, figsize=(12, 18), sharex=True)

    for col in range(active_power_df.shape[1]):
        # smoothed = savgol_filter(active_power_df[col], window_length=51, polyorder=3)
        axs[col].plot(active_power_df[col], linewidth=PLOT_CONFIG["line_width"])
        if col == 0:
            axs[col].set_ylabel("Wind B", fontsize=PLOT_CONFIG["tick_labelsize"])
            axs[col].set_ylim(0, 20)
        elif col == 1:
            axs[col].set_ylabel("Wind C", fontsize=PLOT_CONFIG["tick_labelsize"])
            axs[col].set_ylim(0, 40)
        elif col == 2:
            axs[col].set_ylabel("Wind C-D 1", fontsize=PLOT_CONFIG["tick_labelsize"])
            axs[col].set_ylim(0, 40)
        elif col == 3:
            axs[col].set_ylabel("Wind C-D 2", fontsize=PLOT_CONFIG["tick_labelsize"])
            axs[col].set_ylim(0, 25)
        elif col == 4:
            axs[col].set_ylabel("Wind C-D 3", fontsize=PLOT_CONFIG["tick_labelsize"])
            axs[col].set_ylim(0, 15)
        elif col == 5:
            axs[col].set_ylabel("Wind C-D 4", fontsize=PLOT_CONFIG["tick_labelsize"])
            axs[col].set_ylim(0, 55)
        elif col == 6:
            axs[col].set_ylabel("Wind C-D 5", fontsize=PLOT_CONFIG["tick_labelsize"])
            axs[col].set_ylim(0, 10)

        # axs[col].legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
        axs[col].grid(True)
        axs[col].tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])

    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.tight_layout()
    plt.savefig("figure/wind_active_power.pdf", format="pdf")
    plt.show()


# Example usage:
# plot_wind_activepower("timeseries_results_reference")


def plot_wind_activepower_shorter_period(base_path):
    # the detail is not kept if the plot is possible to read, but if it is possible to read, the peak the vally detail is lost!
    active_power_df = pd.DataFrame()
    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "res_sgen")

        if not os.path.exists(folder_path):
            break  # Stop when a results_* folder is missing

        excel_path = os.path.join(folder_path, "p_mw.xlsx")
        if os.path.exists(excel_path):
            # Read the Excel file
            df = pd.read_excel(excel_path, index_col=0)
            active_power_df = pd.concat([active_power_df, df], ignore_index=True)
        i += 1
    if active_power_df.empty:
        print("NO Data Found")
        return
    active_power_df = active_power_df.iloc[:1441]
    num_points = len(active_power_df)
    xticks = list(range(0, num_points, 144))  # Mark every 1 days
    xticklabels = [f"{i//144}d" for i in xticks]
    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    for col in range(active_power_df.shape[1]):
        smoothed = savgol_filter(active_power_df[col], window_length=51, polyorder=3)
        if col == 0:
            plt.plot(
                smoothed,
                label="Wind B",
                linewidth=PLOT_CONFIG["line_width"],
            )
        elif col == 1:
            plt.plot(
                smoothed,
                label="Wind C",
                linewidth=PLOT_CONFIG["line_width"],
            )
        elif col == 2:
            plt.plot(
                smoothed,
                label="Wind C-D 1",
                linewidth=PLOT_CONFIG["line_width"],
            )
        elif col == 3:
            plt.plot(
                smoothed,
                label="Wind C-D 2",
                linewidth=PLOT_CONFIG["line_width"],
            )
        elif col == 4:
            plt.plot(
                smoothed,
                label="Wind C-D 3",
                linewidth=PLOT_CONFIG["line_width"],
            )
        elif col == 5:
            plt.plot(
                smoothed,
                label="Wind C-D 4",
                linewidth=PLOT_CONFIG["line_width"],
            )
        elif col == 6:
            plt.plot(
                smoothed,
                label="Wind C-D 5",
                linewidth=PLOT_CONFIG["line_width"],
            )
    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Wind Farm Active Power (MW)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    plt.grid(True)
    plt.savefig("figure/wind_active_power_shorter_period.pdf", format="pdf")
    plt.show()


plot_wind_activepower_shorter_period("timeseries_results_reference")


def plot_load_activepower(base_path, frac=0.02):
    # just plot one load is enough
    first_load_df = pd.DataFrame()
    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "res_load")

        if not os.path.exists(folder_path):
            break  # Stop when a results_* folder is missing

        excel_path = os.path.join(folder_path, "p_mw.xlsx")
        if os.path.exists(excel_path):
            # Read the Excel file
            df_p = pd.read_excel(os.path.join(folder_path, "p_mw.xlsx"), index_col=0)
            df_q = pd.read_excel(os.path.join(folder_path, "q_mvar.xlsx"), index_col=0)
            temp_df = pd.DataFrame(
                {
                    "Active Power (MW)": df_p.iloc[:, 0],  # First column from P data
                    "Reactive Power (MVar)": df_q.iloc[
                        :, 0
                    ],  # First column from Q data
                }
            )
            first_load_df = pd.concat([first_load_df, temp_df])

        i += 1
    if first_load_df.empty:
        print("NO Data Found")
        return

    num_points = first_load_df.shape[0]
    xticks = list(range(0, num_points, 4320))  # Mark every 30 days
    xticklabels = [f"{i//144}d" for i in xticks]

    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    for col in range(first_load_df.shape[1]):
        # smoothed = lowess(
        #     active_power_df[col],
        #     np.arange(len(active_power_df)),
        #     frac=frac,
        #     return_sorted=False,
        # )
        smoothed = savgol_filter(
            first_load_df.iloc[:, col], window_length=51, polyorder=3
        )
        # temp = pd.Series(temp)
        # smoothed = temp.ewm(span=40, adjust=False).mean()  # Light smoothing with EMA
        # smoothed = lowess(
        #     first_load_df.iloc[:, col],
        #     np.arange(num_points),
        #     frac=0.02,
        #     return_sorted=False,
        # )
        match col:
            case 0:
                plt.plot(
                    first_load_df.iloc[:, col],
                    label="Wind B",
                    linewidth=PLOT_CONFIG["line_width"],
                )
                plt.plot(smoothed, label="Active power")
            case 1:
                plt.plot(
                    first_load_df.iloc[:, col],
                    label="Wind C",
                    linewidth=PLOT_CONFIG["line_width"],
                )
                plt.plot(smoothed, label="Reactive power")

    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Wind Farm Active Power (MW)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    plt.grid(True)
    plt.show()


# # # Example usage:
# plot_load_activepower("timeseries_results", frac=0.02)


def plot_fault_device_distribution_bar(base_path):
    """
    Reads all results folders inside base_path, extracts fault device distributions, and plots them.
    """
    fault_device_distribution = pd.DataFrame()

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "custom")

        if not os.path.exists(folder_path):
            break  # Stop when a results_* folder is missing

        excel_path = os.path.join(folder_path, "Value.xlsx")
        if os.path.exists(excel_path):
            # Read the Excel file
            df = pd.read_excel(excel_path)
            fault_device_distribution = pd.concat(
                [
                    fault_device_distribution,
                    df[
                        [
                            "Device 0",
                            "Device 1",
                            "Device 2",
                            "Device 3",
                            "Device 4",
                            "Device 5",
                            "Device 6",
                            "Device 7",
                            "Device 8",
                            "Device 9",
                            "Device 10",
                            "Device 11",
                            "Device 12",
                            "Device 13",
                            "Device 14",
                            "Device 15",
                            "Device 16",
                            "Device 17",
                        ]
                    ],
                ],
                ignore_index=True,
            )

        i += 1  # Move to the next results_* folder

    if fault_device_distribution.empty:
        print("NO Data Found")
        return

    # Calculate the average fault count for each device
    avg_faults = fault_device_distribution.mean()

    # Plot the bar chart
    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    avg_faults.plot(kind="bar", color="skyblue")
    plt.xlabel("Device Number", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Average Fault Count", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.xticks(
        ticks=range(len(avg_faults)),
        labels=[f"Device {i}" for i in range(len(avg_faults))],
        rotation=45,
        fontsize=PLOT_CONFIG["tick_labelsize"],
    )

    # Add the average fault count above each bar
    for index, value in enumerate(avg_faults):
        plt.text(
            index,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=PLOT_CONFIG["tick_labelsize"],
        )
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.grid(True)
    plt.show()


# plot_fault_device_distribution_bar("timeseries_results")


def plot_fault_without_gen(base_path_with_gen, base_path_without_gen):
    # Reads all results folders inside base_path_with_gen and base_path_without_gen, extracts fault case ratios, and plots them.
    fault_ratio_comparison = pd.DataFrame(
        columns=["ratio_with_gen", "ratio_without_gen"]
    )

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path_with_gen = os.path.join(
            base_path_with_gen, folder_name, "custom", "Value.xlsx"
        )
        folder_path_without_gen = os.path.join(
            base_path_without_gen, folder_name, "custom", "Value.xlsx"
        )
        if not (
            os.path.exists(folder_path_with_gen)
            or os.path.exists(folder_path_without_gen)
        ):
            break  # Stop when a results_* folder is missing

        # Read the Excel file
        df_with_gen = pd.read_excel(
            folder_path_with_gen, usecols=["Total_fault_cases", "Total_cases_analyzed"]
        )
        df_without_gen = pd.read_excel(
            folder_path_without_gen,
            usecols=["Total_fault_cases", "Total_cases_analyzed"],
        )

        # Store all valid ratios from this file
        fault_ratio_comparison = pd.concat(
            [
                fault_ratio_comparison,
                pd.DataFrame(
                    {
                        "ratio_with_gen": (
                            df_with_gen["Total_fault_cases"]
                            / df_with_gen["Total_cases_analyzed"]
                        ).replace([float("inf"), -float("inf")], None),
                        "ratio_without_gen": (
                            df_without_gen["Total_fault_cases"]
                            / df_without_gen["Total_cases_analyzed"]
                        ).replace([float("inf"), -float("inf")], None),
                    }
                ),
            ],
            ignore_index=True,
        )

        i += 1  # Move to the next results_* folder

    if fault_ratio_comparison.empty:
        print("NO Data Found")
        return

    num_points = fault_ratio_comparison.shape[0]
    xticks = list(range(0, num_points, 4320))  # Mark every 30 days
    xticklabels = [f"{i//144}d" for i in xticks]

    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    plt.plot(
        fault_ratio_comparison["ratio_with_gen"],
        label="With Gen",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.plot(
        fault_ratio_comparison["ratio_without_gen"],
        label="Without Gen",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    plt.grid(True)
    plt.show()


# plot_fault_without_gen("timeseries_results_test", "timeseries_results_test2")


def plot_fault_primaryfault_bar(
    base_path_with_gen, base_path_without_gen, base_path_with_weakexgrid
):
    # Reads all results folders inside base_path_with_gen, base_path_without_gen, and base_path_with_weakexgrid, extracts fault case ratios, and plots them.
    fault_ratio_comparison = pd.DataFrame(
        columns=[
            "primary_with_gen",
            "primary_without_gen",
            "primary_with_weakexgrid",
            "backup_with_gen",
            "backup_without_gen",
            "backup_with_weakexgrid",
        ]
    )

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path_with_gen = os.path.join(
            base_path_with_gen, folder_name, "custom", "Value.xlsx"
        )
        folder_path_without_gen = os.path.join(
            base_path_without_gen, folder_name, "custom", "Value.xlsx"
        )
        folder_path_with_weakexgrid = os.path.join(
            base_path_with_weakexgrid, folder_name, "custom", "Value.xlsx"
        )
        if not (
            os.path.exists(folder_path_with_gen)
            or os.path.exists(folder_path_without_gen)
            or os.path.exists(folder_path_with_weakexgrid)
        ):
            break  # Stop when a results_* folder is missing

        # Read the Excel file
        df_with_gen = pd.read_excel(folder_path_with_gen)
        df_without_gen = pd.read_excel(folder_path_without_gen)
        df_with_weakexgrid = pd.read_excel(folder_path_with_weakexgrid)

        # Store all valid ratios from this file
        fault_ratio_comparison = pd.concat(
            [
                fault_ratio_comparison,
                pd.DataFrame(
                    {
                        "primary_with_gen": df_with_gen["Primary_fail_cases"],
                        "primary_without_gen": df_without_gen["Primary_fail_cases"],
                        "primary_with_weakexgrid": df_with_weakexgrid[
                            "Primary_fail_cases"
                        ],
                        "backup_with_gen": df_with_gen["Backup_fail_cases"],
                        "backup_without_gen": df_without_gen["Backup_fail_cases"],
                        "backup_with_weakexgrid": df_with_weakexgrid[
                            "Backup_fail_cases"
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

        i += 1  # Move to the next results_* folder

    if fault_ratio_comparison.empty:
        print("NO Data Found")
        return

    avg_faults = fault_ratio_comparison.mean()
    avg_faults = pd.DataFrame(
        {
            "with_gen": [avg_faults["primary_with_gen"], avg_faults["backup_with_gen"]],
            "without_gen": [
                avg_faults["primary_without_gen"],
                avg_faults["backup_without_gen"],
            ],
            "with_weakexgrid": [
                avg_faults["primary_with_weakexgrid"],
                avg_faults["backup_with_weakexgrid"],
            ],
        },
        index=["primary", "backup"],
    )

    # Create the plot
    ax = avg_faults.plot(
        kind="bar",
        color={
            "with_gen": "skyblue",
            "without_gen": "salmon",
            "with_weakexgrid": "lightgreen",
        },  # Distinct colors
        edgecolor="black",
        figsize=PLOT_CONFIG["fig_size"],
        legend=True,  # Show legend
    )

    # Add text annotations above the bars
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            label_type="edge",
            fontsize=PLOT_CONFIG["tick_labelsize"],
        )

    # Formatting
    plt.xticks(rotation=0, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Fault Type", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Average Fault Count", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.grid(True)
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    plt.tight_layout()  # Ensure labels fit
    plt.show()


# plot_fault_primaryfault_bar(
#     "timeseries_results_test", "timeseries_results_test2", "timeseries_results_test3"
# )


def plot_fault_overunderreach_bar(
    base_path_with_gen, base_path_without_gen, base_path_with_weakexgrid
):
    # Reads all results folders inside base_path_with_gen, base_path_without_gen, and base_path_with_weakexgrid, extracts overreach and underreach errors, and plots them.
    reach_error_comparison = pd.DataFrame(
        columns=[
            "overreach_with_gen",
            "overreach_without_gen",
            "overreach_with_weakexgrid",
            "underreach_with_gen",
            "underreach_without_gen",
            "underreach_with_weakexgrid",
        ]
    )

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path_with_gen = os.path.join(
            base_path_with_gen, folder_name, "custom", "Value.xlsx"
        )
        folder_path_without_gen = os.path.join(
            base_path_without_gen, folder_name, "custom", "Value.xlsx"
        )
        folder_path_with_weakexgrid = os.path.join(
            base_path_with_weakexgrid, folder_name, "custom", "Value.xlsx"
        )
        if not (
            os.path.exists(folder_path_with_gen)
            or os.path.exists(folder_path_without_gen)
            or os.path.exists(folder_path_with_weakexgrid)
        ):
            break  # Stop when a results_* folder is missing

        # Read the Excel file
        df_with_gen = pd.read_excel(folder_path_with_gen)
        df_without_gen = pd.read_excel(folder_path_without_gen)
        df_with_weakexgrid = pd.read_excel(folder_path_with_weakexgrid)

        # Store all valid errors from this file
        reach_error_comparison = pd.concat(
            [
                reach_error_comparison,
                pd.DataFrame(
                    {
                        "overreach_with_gen": df_with_gen["Overreach_cases"],
                        "overreach_without_gen": df_without_gen["Overreach_cases"],
                        "overreach_with_weakexgrid": df_with_weakexgrid[
                            "Overreach_cases"
                        ],
                        "underreach_with_gen": df_with_gen["Underreach_cases"],
                        "underreach_without_gen": df_without_gen["Underreach_cases"],
                        "underreach_with_weakexgrid": df_with_weakexgrid[
                            "Underreach_cases"
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

        i += 1  # Move to the next results_* folder

    if reach_error_comparison.empty:
        print("NO Data Found")
        return

    avg_errors = reach_error_comparison.mean()
    avg_errors = pd.DataFrame(
        {
            "with_gen": [
                avg_errors["overreach_with_gen"],
                avg_errors["underreach_with_gen"],
            ],
            "without_gen": [
                avg_errors["overreach_without_gen"],
                avg_errors["underreach_without_gen"],
            ],
            "with_weakexgrid": [
                avg_errors["overreach_with_weakexgrid"],
                avg_errors["underreach_with_weakexgrid"],
            ],
        },
        index=["overreach", "underreach"],
    )

    # Create the plot
    ax = avg_errors.plot(
        kind="bar",
        color={
            "with_gen": "skyblue",
            "without_gen": "salmon",
            "with_weakexgrid": "lightgreen",
        },  # Distinct colors
        edgecolor="black",
        figsize=PLOT_CONFIG["fig_size"],
        legend=True,  # Show legend
    )

    # Add text annotations above the bars
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            label_type="edge",
            fontsize=PLOT_CONFIG["tick_labelsize"],
        )

    # Formatting
    plt.xticks(rotation=0, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Error Type", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Average Error Count", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.grid(True)
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    plt.tight_layout()  # Ensure labels fit
    plt.show()


# plot_fault_overunderreach_bar(
#     "timeseries_results_test", "timeseries_results_test2", "timeseries_results_test3"
# )


def plot_fault_with_optimization(base_path, base_path_with_optimization):
    # Reads all results folders inside base_path_with_gen and base_path_without_gen, extracts fault case ratios, and plots them.
    fault_ratio_comparison = pd.DataFrame(
        columns=["ratio_original", "ratio_with_optimization"]
    )

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path_original = os.path.join(
            base_path, folder_name, "custom", "Value.xlsx"
        )
        folder_path_with_optimization = os.path.join(
            base_path_with_optimization, folder_name, "custom", "Value.xlsx"
        )
        if not (
            os.path.exists(folder_path_original)
            or os.path.exists(folder_path_with_optimization)
        ):
            break  # Stop when a results_* folder is missing

        # Read the Excel file
        df_original = pd.read_excel(
            folder_path_original, usecols=["Total_fault_cases", "Total_cases_analyzed"]
        )
        df_with_optimization = pd.read_excel(
            folder_path_with_optimization,
            usecols=["Total_fault_cases", "Total_cases_analyzed"],
        )

        # Store all valid ratios from this file
        fault_ratio_comparison = pd.concat(
            [
                fault_ratio_comparison,
                pd.DataFrame(
                    {
                        "ratio_original": (
                            df_original["Total_fault_cases"]
                            / df_original["Total_cases_analyzed"]
                        ).replace([float("inf"), -float("inf")], None),
                        "ratio_with_optimization": (
                            df_with_optimization["Total_fault_cases"]
                            / df_with_optimization["Total_cases_analyzed"]
                        ).replace([float("inf"), -float("inf")], None),
                    }
                ),
            ],
            ignore_index=True,
        )

        i += 1  # Move to the next results_* folder

    if fault_ratio_comparison.empty:
        print("NO Data Found")
        return

    num_points = fault_ratio_comparison.shape[0]
    xticks = list(range(0, num_points, 4320))  # Mark every 30 days
    xticklabels = [f"{i//144}d" for i in xticks]

    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    plt.plot(
        fault_ratio_comparison["ratio_original"],
        label="Without Optimization",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.plot(
        fault_ratio_comparison["ratio_with_optimization"],
        label="With Optimization",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    plt.grid(True)
    plt.show()


# plot_fault_with_optimization("timeseries_results_test", "timeseries_results_test2")


def plot_fault_with_weak_exgrid(base_path, base_path_weak_exgrid):
    # Reads all results folders inside base_path_with_gen and base_path_without_gen, extracts fault case ratios, and plots them.
    fault_ratio_comparison = pd.DataFrame(
        columns=["ratio_original", "ratio_with_weak_exgrid"]
    )

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path_original = os.path.join(
            base_path, folder_name, "custom", "Value.xlsx"
        )
        folder_path_with_weak_exgrid = os.path.join(
            base_path_weak_exgrid, folder_name, "custom", "Value.xlsx"
        )
        if not (
            os.path.exists(folder_path_original)
            or os.path.exists(folder_path_with_weak_exgrid)
        ):
            break  # Stop when a results_* folder is missing

        # Read the Excel file
        df_original = pd.read_excel(
            folder_path_original, usecols=["Total_fault_cases", "Total_cases_analyzed"]
        )
        df_with_weak_exgrid = pd.read_excel(
            folder_path_with_weak_exgrid,
            usecols=["Total_fault_cases", "Total_cases_analyzed"],
        )

        # Store all valid ratios from this file
        fault_ratio_comparison = pd.concat(
            [
                fault_ratio_comparison,
                pd.DataFrame(
                    {
                        "ratio_original": (
                            df_original["Total_fault_cases"]
                            / df_original["Total_cases_analyzed"]
                        ).replace([float("inf"), -float("inf")], None),
                        "ratio_with_weak_exgrid": (
                            df_with_weak_exgrid["Total_fault_cases"]
                            / df_with_weak_exgrid["Total_cases_analyzed"]
                        ).replace([float("inf"), -float("inf")], None),
                    }
                ),
            ],
            ignore_index=True,
        )

        i += 1  # Move to the next results_* folder

    if fault_ratio_comparison.empty:
        print("NO Data Found")
        return

    num_points = fault_ratio_comparison.shape[0]
    xticks = list(range(0, num_points, 4320))  # Mark every 30 days
    xticklabels = [f"{i//144}d" for i in xticks]

    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    plt.plot(
        fault_ratio_comparison["ratio_original"],
        label="With Strong Exgrid",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.plot(
        fault_ratio_comparison["ratio_with_weak_exgrid"],
        label="With Weak Exgrid",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    plt.grid(True)
    plt.show()


# plot_fault_with_weak_exgrid("timeseries_results_test", "timeseries_results_test2")


def plot_fault_compareload_underreach(base_path):
    # Plot the underreach and overreach cases for anaylzing load and wid generation impact

    fault_comparison = pd.DataFrame(columns=["Ratio", "underreach", "overreach"])

    i = 0  # Start with results_0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "custom")

        if not os.path.exists(folder_path):
            break  # Stop when a results_* folder is missing

        excel_path = os.path.join(folder_path, "Value.xlsx")
        if os.path.exists(excel_path):
            # Read the Excel file
            df = pd.read_excel(
                excel_path,
                usecols=[
                    "Total_fault_cases",
                    "Total_cases_analyzed",
                    "Overreach_cases",
                    "Underreach_cases",
                ],
            )

            if (
                "Total_fault_cases" in df.columns
                and "Total_cases_analyzed" in df.columns
                and "Overreach_cases" in df.columns
                and "Underreach_cases" in df.columns
            ):
                # Compute the row-wise ratio, avoiding division by zero
                df["Ratio"] = df["Total_fault_cases"] / df["Total_cases_analyzed"]
                df["Ratio"] = df["Ratio"].replace(
                    [float("inf"), -float("inf")], None
                )  # Handle divide-by-zero cases

                # Store all valid ratios from this file
                fault_comparison = pd.concat(
                    [
                        fault_comparison,
                        df[["Ratio", "Underreach_cases", "Overreach_cases"]].dropna(),
                    ],
                    ignore_index=True,
                )

        i += 1  # Move to the next results_* folder
    # Wind Generation Data
    active_power_df = pd.DataFrame()
    i = 0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "res_sgen")
        if not os.path.exists(folder_path):
            break
        excel_path = os.path.join(folder_path, "p_mw.xlsx")
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, index_col=0)
            active_power_df = pd.concat([active_power_df, df], ignore_index=True)
        i += 1

    # Load Data (Active and Reactive Power)
    first_load_df = pd.DataFrame()
    i = 0
    while True:
        folder_name = f"results_{i}"
        folder_path = os.path.join(base_path, folder_name, "res_load")
        if not os.path.exists(folder_path):
            break
        excel_path_p = os.path.join(folder_path, "p_mw.xlsx")
        excel_path_q = os.path.join(folder_path, "q_mvar.xlsx")
        if os.path.exists(excel_path_p) and os.path.exists(excel_path_q):
            df_p = pd.read_excel(excel_path_p, index_col=0)
            df_q = pd.read_excel(excel_path_q, index_col=0)
            temp_df = pd.DataFrame(
                {
                    "Active Power": df_p.iloc[:, 0],
                    "Reactive Power": df_q.iloc[:, 0],
                }
            )
            first_load_df = pd.concat([first_load_df, temp_df])
        i += 1

    number_of_points = 721  # 5 days
    xticks = list(range(0, number_of_points, 144))  # Mark every 1 days
    xticklabels = [f"{i//144}d" for i in xticks]
    # truncate the data
    active_power_df = active_power_df.iloc[:number_of_points]
    first_load_df = first_load_df.iloc[:number_of_points]
    fault_comparison = fault_comparison.iloc[:number_of_points]

    # Plot the results
    fig, axs = plt.subplots(3, 1, figsize=PLOT_CONFIG["fig_large_size"], sharex=True)
    # Underreach and Overreach Cases Plot
    min_under = fault_comparison["Underreach_cases"].min()
    max_under = fault_comparison["Underreach_cases"].max()
    min_over = fault_comparison["Overreach_cases"].min()
    max_over = fault_comparison["Overreach_cases"].max()

    axs[0].plot(
        fault_comparison["Underreach_cases"],
        label="Underreach Cases",
        color="#0066cc",
        linewidth=PLOT_CONFIG["line_width"],
    )
    axs[0].set_ylabel("Underreach Cases", fontsize=PLOT_CONFIG["axis_labelsize"])
    axs[0].tick_params(axis="y")
    axs[0].axvline(
        x=93, color="red", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[0].axvline(
        x=291, color="red", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[0].text(192, 2200, "\u2460", color="red", fontsize=20, ha="center")  # ①

    axs[0].axvline(
        x=353, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[0].axvline(
        x=476, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[0].text(414, 2200, "\u2461", color="green", fontsize=20, ha="center")  # ②
    axs[0].set_ylim(2100, 2300)
    axs[0].tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])

    ax2 = axs[0].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(
        fault_comparison["Overreach_cases"],
        label="Overreach Cases",
        color="darkorange",
        linewidth=PLOT_CONFIG["line_width"],
    )
    ax2.set_ylim(0, 25)
    ax2.set_ylabel("Overreach Cases", fontsize=PLOT_CONFIG["axis_labelsize"])
    ax2.tick_params(axis="y")

    axs[0].legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    ax2.legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="lower right")

    # Wind Generation Plot
    axs[1].plot(
        active_power_df.index,
        active_power_df.iloc[:, 0],
        label="Active Power (MW)",
        color="#0066cc",
        linewidth=PLOT_CONFIG["line_width"],
    )
    axs[1].axvline(
        x=93, color="red", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[1].axvline(
        x=291, color="red", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[1].axvline(
        x=353, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[1].axvline(
        x=476, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    # axs[1].axvline(
    #     x=343, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    # )
    # axs[1].axvline(
    #     x=472, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    # )
    axs[1].set_ylabel(
        "Wind B Active Power (MW)", fontsize=PLOT_CONFIG["axis_labelsize"]
    )
    axs[1].tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])
    # axs[1].legend(fontsize=PLOT_CONFIG["legend_fontsize"])

    # Load Data Plot
    axs[2].plot(
        first_load_df.index,
        first_load_df["Active Power"],
        label="Active Power",
        color="#0066cc",
        linewidth=PLOT_CONFIG["line_width"],
    )
    # axs[2].plot(
    #     first_load_df.index,
    #     first_load_df["Reactive Power"],
    #     label="Reactive Power",
    #     color="dimgrey",
    #     linewidth=PLOT_CONFIG["line_width"],
    # )
    axs[2].axvline(
        x=93, color="red", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[2].axvline(
        x=291, color="red", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[2].axvline(
        x=353, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[2].axvline(
        x=476, color="green", linestyle="--", linewidth=PLOT_CONFIG["line_width"]
    )
    axs[2].set_ylabel("Load Active Power (MW)", fontsize=PLOT_CONFIG["axis_labelsize"])
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    axs[2].tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])
    # axs[2].legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")

    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.tight_layout()
    plt.savefig("fault_compareload_underreach.pdf")
    plt.show()


# plot_fault_compareload_underreach("timeseries_results_reference")
