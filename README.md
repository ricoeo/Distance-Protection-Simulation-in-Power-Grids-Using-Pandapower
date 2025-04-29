# Power Grid Simulation and Fault Analysis

Welcome to the **Probabilistic Analysis of Distance Protection Accuracy in Power Grids Using Pandapower** project! This repository contains tools and scripts for simulating power grid dynamics, analyzing fault scenarios, and visualizing results. The project is designed to analyze the performance of pair-installed distance protection devices in the AC grid. 

---

## 🚀 Features

- **Distance Protection Power Grid Simulation**: Scan the network topology and implement distance protection logic. Conduct short-circuit simulation at each line.
- **Simulation on real and one-year-long profiles**: Perform year-long simulations with dynamic load power and wind power profile (110kV) and also periodic result saving.
- **Fault Detection and Protection**: (optional) Adjust protection zones dynamically based on real-time measurement data.
- **Custom Visualization**: Generate insightful plots for load power, wind power, and short-circuit initial power flow.
- **Modular Design**: Easily extendable with shared libraries for network creation, result processing, and profile extraction.

---

## 📂 Project Structure

### Key Files and Directories

1. **`timeseries_combine_fault_check.py`**  
   - Main script for running the time-series simulation.
   - Implements dynamic profile updates and fault simulation.

2. **`shared/`**  
   - Contains reusable modules for network creation, profile extraction, protection zone adjustments, and result processing:
     - `library.py`: Core utilities for creating and managing the power grid network.
     - `extract_profile.py`: Handles dynamic profile loading.
     - `protection_border_adjust.py`: Adjusts protection zones based on measurement data.
     - `result_processing.py`: Tools for visualizing and processing simulation results.

3. **`grid_data_sheet.xlsx`**  
   - Excel file containing the grid configuration data. (Example grid configurations from the Disego project)

4. **`fault_detection_check_BE_AB1.xlsx`**  
   - Static measurement data for fault evaulation and optimization.

5. **`simulation_results/`**  
   - Directory where simulation results are saved in `.xlsx` format.

6. **`external zones/`**  
   - Contains external zone settings for comparison purposes. (If a well-tuned distance protection parameters are formed, adjust the format to the example excel)

7. **`sc_pf_half_position/`**  
   - Stores short-circuit analysis plots. (For my usage of analyzing the fault current directrion and magnitude)

8. **`thesis/`**  
   - Supporting documents and resources related to the project.


---

## 🛠️ Setup

1. **Clone the Repository**
2. **Install Dependencies**: Python environment with pandas, pandapower, simbench, wetterdiesnt, numpy, openpyxl, matplotlib, scipy
3. **Prepare Input Data**:Place your grid configuration in grid_data_sheet.xlsx.

---
## ▶️ Usage
1. **Static distance proteciton performance**
   
   **check the result**: please refer the file "fault_detection_check.py"
   
   **result-based optimization (optional)**: please refer the file "test (not test cases)\adjust_protection_border.py" for usage and "shared\protection_border_adjust.py" for the algorithm
   
3. **Dynamic distance protection performance**

   **simulate the result (it may take over 1 month)**: please refer the file "timeseries_combine_fault_check.py" 
   
   **exchange the profile source**: please refer the file "shared\extract_profile.py"

---
## 📈 Example Outputs
Fault type and amount accross devices: [fault_device_distribution.pdf](https://github.com/user-attachments/files/19959712/fault_device_distribution_with_primary.pdf)

Short-circuit initial current flow: [Initial_current_flow_strong_grid.pdf](https://github.com/user-attachments/files/19959677/MA_Zhao_Ming_0423.Extract.68.pdf)

---
## 🤝 Contributing
Feel free to improve the project, improvements to this project are not in the author's plans.

---
## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

