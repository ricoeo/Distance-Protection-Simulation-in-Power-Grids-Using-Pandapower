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
    "tick_labelsize": 16,  # Font size for axis ticks
    "line_width": 2,  # Line width for plots
    "marker_size": 6,  # Marker size for highlighted points
    "value_fontsize": 14,  # Font size for value annotations
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

    ratios = ratios[:-145]
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
    # Show tick marks for all weeks
    xticks = range(1, len(week_labels) + 1)  # Tick positions for all weeks

    # Create labels: display only for every 2 weeks, empty strings for others
    xticklabels = [
        week_labels[i] if (i + 1) % 3 == 1 else "" for i in range(len(week_labels))
    ]

    # Set ticks and labels
    plt.xticks(
        ticks=xticks,
        labels=xticklabels,
        ha="center",
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


# plot_wind_activepower_shorter_period("timeseries_results_reference")


def plot_load_power(base_path, frac=0.02):
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
                    "Active Power": df_p.iloc[:, 0],  # First column from P data
                    "Reactive Power": df_q.iloc[:, 0],  # First column from Q data
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

    fig, ax1 = plt.subplots(figsize=(12, 8))
    min_p = first_load_df["Active Power"].min()
    max_p = first_load_df["Active Power"].max()
    min_q = first_load_df["Reactive Power"].min()
    max_q = first_load_df["Reactive Power"].max()

    smoothed_active = lowess(
        first_load_df["Active Power"],
        np.arange(num_points),
        frac=frac,
        return_sorted=False,
    )
    smoothed_reactive = lowess(
        first_load_df["Reactive Power"],
        np.arange(num_points),
        frac=frac,
        return_sorted=False,
    )

    ax1.plot(
        first_load_df["Active Power"],
        label="Active Power",
        linewidth=PLOT_CONFIG["line_width"],
    )
    ax1.plot(smoothed_active, label="Smoothed Active Power", linestyle="--")
    ax1.set_ylabel("Active Power (MW)", fontsize=PLOT_CONFIG["axis_labelsize"])
    ax1.tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])
    ax1.set_ylim(min_p - (max_p - min_p), max_p * 1.1)

    ax2 = ax1.twinx()
    ax2.plot(
        first_load_df["Reactive Power"],
        label="Reactive Power",
        linewidth=PLOT_CONFIG["line_width"],
        color="#ffe44eff",
    )
    ax2.plot(
        smoothed_reactive,
        label="Smoothed Reactive Power",
        linestyle="--",
        color="#ff0000ff",
    )
    ax2.set_ylabel("Reactive Power (MVAR)", fontsize=PLOT_CONFIG["axis_labelsize"])
    ax2.tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])
    ax2.set_ylim(min_q, max_q + (max_q - min_q) * 2)

    fig.tight_layout()
    fig.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    ax1.grid(True)

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    ax1.set_xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    fig.tight_layout()
    fig.savefig("figure/load_power.pdf", format="pdf")
    plt.show()


# Example usage:
# plot_load_power("timeseries_results_reference", frac=0.01)


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
    avg_faults.plot(kind="bar", edgecolor="black")
    for y in range(50, 450, 50):
        plt.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, zorder=0)
    plt.xlabel("Device Number", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Average Fault Count", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.xticks(
        ticks=range(len(avg_faults)),
        labels=[f"{i}" for i in range(len(avg_faults))],
        rotation=0,
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
            fontsize=PLOT_CONFIG["value_fontsize"],
        )
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.tight_layout()
    plt.savefig("figure/fault_device_distribution.pdf", format="pdf")
    # plt.grid(True)
    plt.show()


# plot_fault_device_distribution_bar("timeseries_results_reference")


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
            and os.path.exists(folder_path_without_gen)
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

        # Ensure df_with_gen has the same number of rows as df_without_gen, because the simualtion duration of wihout gen case is one month
        if df_with_gen.shape[0] > df_without_gen.shape[0]:
            df_with_gen = df_with_gen.iloc[: df_without_gen.shape[0]]

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
    xticks = list(range(0, num_points + 1, 720))  # Mark every 30 days
    xticklabels = [f"{(i+1)//144}d" for i in xticks]

    fig, axs = plt.subplots(2, 1, figsize=PLOT_CONFIG["fig_size"], sharex=True)
    axs[0].plot(
        fault_ratio_comparison["ratio_with_gen"],
        label="Case 1",
        linewidth=PLOT_CONFIG["line_width"],
    )
    axs[0].plot(
        fault_ratio_comparison["ratio_without_gen"],
        label="Case 2",
        linewidth=PLOT_CONFIG["line_width"],
    )
    axs[0].tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])
    axs[0].set_ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    axs[0].legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    axs[0].grid(True)
    # Plot the difference
    axs[1].plot(
        fault_ratio_comparison["ratio_with_gen"]
        - fault_ratio_comparison["ratio_without_gen"],
        label="Difference",
        linewidth=PLOT_CONFIG["line_width"],
        color="#2e8b57ff",
    )
    axs[1].set_ylabel("Difference in Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    axs[1].grid(True)
    # plt.plot(
    #     fault_ratio_comparison["ratio_with_gen"],
    #     label="Case 1",
    #     linewidth=PLOT_CONFIG["line_width"],
    # )
    # plt.plot(
    #     fault_ratio_comparison["ratio_without_gen"],
    #     label="Case 2",
    #     linewidth=PLOT_CONFIG["line_width"],
    # )
    # plt.plot(
    #     fault_ratio_comparison["ratio_with_gen"]
    #     - fault_ratio_comparison["ratio_without_gen"],
    #     label="Difference",
    #     linewidth=PLOT_CONFIG["line_width"],
    # )
    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("figure/fault_without_gen.pdf", format="pdf")
    plt.show()


plot_fault_without_gen("timeseries_results_reference", "timeseries_results_without_gen")


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
            and os.path.exists(folder_path_without_gen)
            and os.path.exists(folder_path_with_weakexgrid)
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
            and os.path.exists(folder_path_without_gen)
            and os.path.exists(folder_path_with_weakexgrid)
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
            and os.path.exists(folder_path_with_optimization)
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
    fault_ratio_comparison = fault_ratio_comparison[:-145]
    # line plot is boring

    # num_points = fault_ratio_comparison.shape[0]
    # xticks = list(range(0, num_points, 4320))  # Mark every 30 days
    # xticklabels = [f"{i//144}d" for i in xticks]

    # plt.figure(figsize=PLOT_CONFIG["fig_size"])
    # plt.plot(
    #     fault_ratio_comparison["ratio_original"],
    #     label="Without Optimization",
    #     linewidth=PLOT_CONFIG["line_width"],
    # )
    # plt.plot(
    #     fault_ratio_comparison["ratio_with_optimization"],
    #     label="With Optimization",
    #     linewidth=PLOT_CONFIG["line_width"],
    # )
    # plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    # plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    # plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    # plt.ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    # plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"])
    # plt.grid(True)
    # plt.show()

    # box plot
    # Group data into weekly bins (1008 data points/week)
    WEEKLY_RESOLUTION = 1008
    fault_ratio_comparison["Week"] = (
        fault_ratio_comparison.index // WEEKLY_RESOLUTION
    ) + 1

    # Prepare data for boxplot
    grouped_original = fault_ratio_comparison.groupby("Week")["ratio_original"].apply(
        list
    )
    grouped_with_optimization = fault_ratio_comparison.groupby("Week")[
        "ratio_with_optimization"
    ].apply(list)
    box_data_original = grouped_original.tolist()
    box_data_with_optimization = grouped_with_optimization.tolist()
    week_labels = [f"{w}" for w in grouped_original.index]

    # Plot the boxplot
    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    positions = range(1, len(week_labels) + 1)

    # Plot original data
    box_original = plt.boxplot(
        box_data_original,
        positions=[p - 0.2 for p in positions],
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        labels=week_labels,
    )

    # Plot optimized data
    box_optimized = plt.boxplot(
        box_data_with_optimization,
        positions=[p + 0.2 for p in positions],
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        labels=week_labels,
    )

    # Configure axes
    # Show labels only at every 2 weeks (e.g., 1, 3, 5, ...)
    # Show tick marks for all weeks
    xticks = range(1, len(week_labels) + 1)  # Tick positions for all weeks

    # Create labels: display only for every 2 weeks, empty strings for others
    xticklabels = [
        week_labels[i] if (i + 1) % 3 == 1 else "" for i in range(len(week_labels))
    ]

    # Set ticks and labels
    plt.xticks(
        ticks=xticks,
        labels=xticklabels,
        ha="center",
        fontsize=PLOT_CONFIG["tick_labelsize"],
    )
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.ylim(0.14, 0.26)
    plt.xlabel("Week Number", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend(
        [box_original["boxes"][0], box_optimized["boxes"][0]],
        ["Case 1", "Case 4"],
        fontsize=PLOT_CONFIG["legend_fontsize"],
    )
    plt.tight_layout()

    # Save the figure as a PDF
    plt.savefig("figure/fault_case_ratios_comparison_optimization.pdf", format="pdf")
    plt.show()


# plot_fault_with_optimization(
#     "timeseries_results_reference", "timeseries_results_new_protection_zone"
# )


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
            and os.path.exists(folder_path_with_weak_exgrid)
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
        # Ensure df_with_gen has the same number of rows as df_without_gen, because the simualtion duration of wihout gen case is one month
        if df_original.shape[0] > df_with_weak_exgrid.shape[0]:
            df_original = df_original.iloc[: df_with_weak_exgrid.shape[0]]
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
    xticks = list(range(0, num_points + 1, 720))  # Mark every 30 days
    xticklabels = [f"{(i+1)//144}d" for i in xticks]

    plt.figure(figsize=PLOT_CONFIG["fig_size"])
    plt.plot(
        fault_ratio_comparison["ratio_original"],
        label="Case 1",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.plot(
        fault_ratio_comparison["ratio_with_weak_exgrid"],
        label="Case 3",
        linewidth=PLOT_CONFIG["line_width"],
    )
    plt.xticks(xticks, xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    plt.ylim(0.23, 0.29)
    plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.ylabel("Fault Case Ratio", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("figure/fault_with_weak_exgrid.pdf", format="pdf")
    plt.show()


# plot_fault_with_weak_exgrid(
#     "timeseries_results_reference", "timeseries_results_weak_exgrid"
# )


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
        "Wind B Active Power (MW)",
        labelpad=22.5,
        fontsize=PLOT_CONFIG["axis_labelsize"],
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
    axs[2].set_ylabel(
        "Load Active Power (MW)", labelpad=14, fontsize=PLOT_CONFIG["axis_labelsize"]
    )
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticklabels, fontsize=PLOT_CONFIG["tick_labelsize"])
    axs[2].set_xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    axs[2].tick_params(axis="y", labelsize=PLOT_CONFIG["tick_labelsize"])
    # axs[2].legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")

    plt.yticks(fontsize=PLOT_CONFIG["tick_labelsize"])
    # plt.xlabel("Time (Days)", fontsize=PLOT_CONFIG["axis_labelsize"])
    plt.tight_layout()
    plt.savefig("figure/fault_compareload_underreach.pdf")
    plt.show()


# plot_fault_compareload_underreach("timeseries_results_reference")
