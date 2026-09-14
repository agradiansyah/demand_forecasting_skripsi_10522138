# Demand Forecasting - The Soko Coffee Tea Chocolate

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

This repository contains the source code for the Undergraduate Thesis (Skripsi) titled **"Forecasting Raw Material Inventory Needs Using Linear Regression and Random Forest at The Soko Coffee Tea Chocolate"**.

## Author Information
*   **Name:** Aqbil Gradiansyah
*   **Student ID (NIM):** 10522138
*   **Study Program:** Information Systems
*   **University:** Universitas Komputer Indonesia (UNIKOM)

## Project Description
This system was built to help **The Soko Coffee Tea Chocolate** predict daily and monthly raw material inventory needs to prevent stockouts and overstocking. 

The project utilizes historical sales data, which is extracted into raw material requirements using the **Bill of Materials (BOM) Explosion** method. The data is then modeled using two Machine Learning algorithms to compare their performance:
1.  **Linear Regression:** Used as a baseline model to identify linear trends in the data movement.
2.  **Random Forest:** Used to capture more complex and non-linear patterns within the data.

## Key Features
*   **Data Preprocessing:** Cleaning sales data and converting menu items into raw material requirements (BOM Explosion).
*   **Forecasting Engine:** Predicting future raw material needs using Linear Regression and Random Forest models.
*   **Model Evaluation:** Comparing accuracy metrics (such as RMSE, MAE, or MAPE) between the two models.
*   **Owner Dashboard (`app_owner.py`):** An interactive interface designed for the business owner to view forecasting results and make data-driven inventory procurement decisions.

## Technologies Used
*   **Programming Language:** Python
*   **Data & ML Libraries:** Pandas, NumPy, Scikit-Learn
*   **Deployment/Dashboard:** Streamlit / Tkinter (via `app_owner.py`)

## How to Run the Program

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/agradiansyah/demand_forecasting_skripsi_10522138.git](https://github.com/agradiansyah/demand_forecasting_skripsi_10522138.git)
    cd demand_forecasting_skripsi_10522138
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have generated a requirements.txt file if you are using a virtual environment)*

3.  **Run the main application:**
    ```bash
    python app_owner.py
    ```
    *(Use `streamlit run app_owner.py` if the dashboard is built using Streamlit)*

## Analysis Conclusion
The evaluation results of this system provide an objective performance comparison between the Linear Regression and Random Forest models, ultimately recommending the most accurate algorithm to be permanently implemented in The Soko Coffee Tea Chocolate's inventory management workflow.

---
*© 2026 Aqbil Gradiansyah. All rights reserved.*
