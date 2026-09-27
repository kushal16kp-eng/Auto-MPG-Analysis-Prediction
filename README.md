# Auto MPG Analysis & Prediction

A data exploration and machine learning project analyzing the classic **Auto MPG dataset** — cleaning the data, exploring patterns through visualizations, and building a Linear Regression model to predict a car's fuel efficiency (MPG) from its features.

> Built as part of the Coding Ninjas 10X AI-ML Recruitment Task.

## 📌 Project Overview

This project is split into two parts:

1. **Exploratory Data Analysis & Preprocessing** — cleaning the raw dataset and exploring relationships between car features and fuel efficiency.
2. **Linear Regression** — building and evaluating a model that predicts MPG from vehicle characteristics.

## 🛠️ Technologies Used

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `Auto_MPG_Analysis_Prediction.ipynb` | Main Google Colab notebook with all code, analysis, and results |
| `auto_mpg_cleaned.csv` | The cleaned, analysis-ready dataset |
| `README.md` | This file |

## 🔍 Task 1: Data Exploration & Preprocessing

- Loaded and inspected the Auto MPG dataset (398 cars, 9 features)
- Handled missing `horsepower` values using median imputation
- Checked for duplicate and invalid records (none found)
- Explored distributions of MPG, weight, horsepower, displacement, and acceleration
- Analyzed correlations between MPG and other variables
- Investigated the question: *"Do cars from the USA, Europe, or Japan tend to have more powerful engines?"*

**Key Findings:**
- Weight has the strongest negative correlation with MPG (**-0.83**)
- Displacement (-0.80), cylinders (-0.78), and horsepower (-0.77) also strongly hurt fuel efficiency
- Model year has a positive correlation with MPG (**+0.58**) — cars became more efficient over time
- American cars have far higher average horsepower (**118.6 hp**) than European (**80.9 hp**) or Japanese (**79.8 hp**) cars

## 🤖 Task 2: Linear Regression

- Prepared features (removed `car_name`, one-hot encoded `origin`)
- Split data 80/20 for training/testing (`random_state=42`)
- Built a single-feature model (weight only) and a multi-feature model (all features)
- Evaluated both using MAE, MSE, RMSE, and R²

**Results:**

| Metric | Single-feature (weight only) | Multi-feature (all) |
|--------|------------------------------|----------------------|
| MAE    | 3.118 | 2.288 |
| RMSE   | 3.859 | 2.888 |
| R²     | 0.723 | 0.845 |

The multi-feature model explains **84.5%** of the variation in MPG, with an average prediction error of only **~2.3 MPG**. Training R² (0.819) and testing R² (0.845) are close together, indicating the model generalizes well with no significant overfitting.

**Limitations:** Linear Regression assumes straight-line relationships, is sensitive to outliers, struggles with correlated features (multicollinearity), and is limited to the features available in the dataset.

**Possible Improvement:** Trying a more flexible model like Random Forest or Gradient Boosting to capture non-linear relationships.

## 🚀 How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run all cells from top to bottom (`Runtime → Run all`)
3. The dataset is loaded automatically from the UCI Machine Learning Repository — no manual download needed

## 👤 Author

Kushal
