import pandas as pd
import math

"""
This module contains the gradient descent algorithm to adjust the protection zone boundaries of devices based on measurement data.
"""


def compute_sensed_impedance(r, a):
    """
    Calculate the imaginary part of the sensed impedance based on r_sensed and angle_sensed.
    """
    imag_sensed = r * math.sin(math.radians(a))
    return imag_sensed


def compute_misjudgment_counts(device_data, boundaries):
    """
    Calculate the number of misjudgments for each zone boundary.
    """
    misjudgment_counts = [
        0,
        0,
        0,
    ]  # For boundaries between Zone 1-2, Zone 2-3, Zone 3-Out of Zone

    for _, row in device_data.iterrows():
        zone_calculated = row["zone_calculated"]
        zone_sensed_img = compute_sensed_impedance(
            row["r_sensed"], row["angle_sensed"]
        )  # recalculate the zone after changing the polygon

        # Determine the sensed zone based on the current boundaries
        if zone_sensed_img < boundaries[0]:
            sensed_zone = "Zone 1"
        elif zone_sensed_img < boundaries[1]:
            sensed_zone = "Zone 2"
        elif zone_sensed_img < boundaries[2]:
            sensed_zone = "Zone 3"
        else:
            sensed_zone = "Out of Zone"

        # Count misjudgments for each boundary based on calculated and sensed zones
        if zone_calculated == "Zone 1" and sensed_zone == "Zone 2":
            misjudgment_counts[0] += 1  # Boundary between Zone 1 and Zone 2
            # up_misjudgment[0] += 1
        elif zone_calculated == "Zone 2" and sensed_zone == "Zone 1":
            misjudgment_counts[0] += 1
            # down_misjudgment[0] += 1

        if zone_calculated == "Zone 2" and sensed_zone == "Zone 3":
            misjudgment_counts[1] += 1  # Boundary between Zone 2 and Zone 3
            # up_misjudgment[1] += 1
        elif zone_calculated == "Zone 3" and sensed_zone == "Zone 2":
            misjudgment_counts[1] += 1
            # down_misjudgment[1] += 1

        if zone_calculated == "Zone 3" and sensed_zone == "Out of Zone":
            misjudgment_counts[2] += 1  # Boundary between Zone 3 and Out of Zone
            # up_misjudgment[2] += 1
        elif zone_calculated == "Out of Zone" and sensed_zone == "Zone 3":
            misjudgment_counts[2] += 1
            # down_misjudgment[2] += 1

    return misjudgment_counts


def finalize_boundaries(boundary_selection_option, original_boundaries):
    # Initialize the final boundaries with original values
    final_boundaries = original_boundaries[:]

    # Iterate over each zone to choose the best boundary
    for zone_index in range(len(original_boundaries)):
        # Get the candidate boundaries that resulted in minimum misjudgment
        candidate_boundaries = boundary_selection_option[zone_index]
        # if there is no change in the boundary, this list gonna be empty, means the previous border setting is already perfect detecting faults
        if not candidate_boundaries:
            continue

        # Sort candidates to ensure order
        candidate_boundaries = sorted(candidate_boundaries)

        # Select a boundary value while keeping zones ordered:
        # Ensure that Zone 1 < Zone 2 < Zone 3.
        if zone_index == 0:
            # For the first zone, take the middle candidate
            middle_index = len(candidate_boundaries) // 2
            final_boundaries[zone_index] = candidate_boundaries[middle_index]
        elif zone_index == len(original_boundaries) - 1:
            # For the last zone, take the largest candidate but ensure it's greater than the previous zone
            valid_candidates = [
                b for b in candidate_boundaries if b > final_boundaries[zone_index - 1]
            ]
            if valid_candidates:
                final_boundaries[zone_index] = max(valid_candidates)
            else:
                # If no valid candidate found, use the original boundary or zone 2 boundary + 0.1
                final_boundaries[zone_index] = max(
                    original_boundaries[zone_index],
                    final_boundaries[zone_index - 1] + 0.1,
                )
        else:
            # For intermediate zones, ensure the selected boundary is greater than the previous zone's boundary
            valid_candidates = [
                b for b in candidate_boundaries if b > final_boundaries[zone_index - 1]
            ]
            if valid_candidates:
                final_boundaries[zone_index] = valid_candidates[0]
            else:
                # If no valid candidate found, fall back to the original boundary value or zone boundary + 0.1
                final_boundaries[zone_index] = max(
                    original_boundaries[zone_index],
                    final_boundaries[zone_index - 1] + 0.1,
                )

    return final_boundaries


def adjust_zone_boundaries(
    device_data, protection_device, initial_step=0.1, max_iterations=100
):
    # Get the initial zone boundaries
    original_boundaries = [z.imag for z in protection_device.associated_zone_impedance]

    # Store previously tested boundaries and their misjudgments for each zone
    boundary_iter_history = {i: {} for i in range(len(original_boundaries))}
    boundary_selection_option = {i: {} for i in range(len(original_boundaries))}

    for zone_index in range(len(original_boundaries)):
        # Skip if the boundary is None, 0, or 0j as no adjustment is needed
        if (
            original_boundaries[zone_index] is None
            or original_boundaries[zone_index] == 0
            or original_boundaries[zone_index] == 0j
        ):
            continue

        boundaries = original_boundaries[:]
        current_boundary = boundaries[zone_index]
        initial_direction = None

        # it is low chance, but when the misjudgements equals to 0, boudary is not needed to be adjusted
        original_misjudgement = compute_misjudgment_counts(
            device_data, original_boundaries
        )
        if original_misjudgement[zone_index] == 0:
            continue

        # Decide the initial adjustment direction based on misjudgments
        boundary_with_disturbance_up = original_boundaries[zone_index] + 0.1
        boundaries[zone_index] = boundary_with_disturbance_up
        sum_misjudgment_1 = compute_misjudgment_counts(device_data, boundaries)
        boundary_with_disturbance_down = original_boundaries[zone_index] - 0.1
        boundaries[zone_index] = boundary_with_disturbance_down
        sum_misjudgment_2 = compute_misjudgment_counts(device_data, boundaries)

        if sum_misjudgment_1[zone_index] < sum_misjudgment_2[zone_index]:
            initial_direction = 1  # Upwards adjustment
            current_boundary = boundary_with_disturbance_up
        else:
            initial_direction = -1  # Downwards adjustment
            current_boundary = boundary_with_disturbance_down

        # store initial misjudgment data
        boundary_iter_history[zone_index][original_boundaries[zone_index]] = (
            original_misjudgement[zone_index]
        )
        boundary_iter_history[zone_index][boundary_with_disturbance_up] = (
            sum_misjudgment_1[zone_index]
        )
        boundary_iter_history[zone_index][boundary_with_disturbance_down] = (
            sum_misjudgment_2[zone_index]
        )
        best_misjudgment = min(
            sum_misjudgment_1[zone_index],
            sum_misjudgment_2[zone_index],
            original_misjudgement[zone_index],
        )

        # Track the direction and cost for the current boundary
        direction = initial_direction
        for _ in range(max_iterations):
            # Adjust boundary in the current direction
            current_boundary += direction * initial_step
            boundaries[zone_index] = (
                current_boundary  # Update boundaries temporarily for evaluation
            )

            # Calculate the new misjudgment for this adjustment
            new_misjudgment = compute_misjudgment_counts(device_data, boundaries)[
                zone_index
            ]
            # Store the new boundary and its cost
            boundary_iter_history[zone_index][current_boundary] = new_misjudgment

            if new_misjudgment < best_misjudgment:
                best_misjudgment = new_misjudgment
            elif new_misjudgment > best_misjudgment:
                direction *= -1  # Switch adjustment direction
                if direction == 1:
                    current_boundary = max(boundary_iter_history[zone_index].keys())
                else:
                    current_boundary = min(boundary_iter_history[zone_index].keys())
            # maybe no need to implement the stop condition but let it run the iteration all

        # Gather all boundaries that have the minimum misjudgment
        min_keys = [
            key
            for key, value in boundary_iter_history[zone_index].items()
            if value == best_misjudgment
        ]
        boundary_selection_option[zone_index] = min_keys
    # Finalize the boundaries to ensure proper ordering and minimal misjudgment
    final_boundaries = finalize_boundaries(
        boundary_selection_option, original_boundaries
    )

    # Update the zone impedances with the adjusted imaginary parts
    protection_device.associated_zone_impedance = [
        (
            complex(z.real * final_boundaries[i] / z.imag, final_boundaries[i])
            if original_boundaries[i] not in (None, 0, 0j)
            else z
        )
        for i, z in enumerate(protection_device.associated_zone_impedance)
    ]

    return protection_device


def adjust_protection_zone_with_measurement_data(measurement_data, protection_devices):
    adjust_zone = {}

    for device in protection_devices:
        adjust_zone[device] = adjust_zone_boundaries(
            measurement_data[measurement_data["Device ID"] == device],
            protection_devices[device],
        )

    return adjust_zone


def adjust_protection_zone_with_external_parameters(excel, protection_devices):
    # in case the excel file has empty entries, it is easier to input an exisitng initialized protection device list
    external_parameters = pd.read_excel(excel)

    for device in protection_devices:
        extern_list = external_parameters.loc[device]
        extern_associated_zone_impedance = [
            complex(extern_list["RE_1"], extern_list["X_1"]),
            complex(extern_list["RE_2"], extern_list["X_2"]),
            complex(extern_list["RE_3"], extern_list["X_3"]),
        ]
        # Update the protection device's associated zone impedance\
        protection_devices[device].associated_zone_impedance = (
            extern_associated_zone_impedance
        )

    return protection_devices
