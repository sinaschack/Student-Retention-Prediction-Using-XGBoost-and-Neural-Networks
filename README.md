# Student Retention Prediction Using XGBoost and Neural Networks

**Predict student dropout using supervised learning with XGBoost and Neural Networks.** This project provides **actionable insights** for early student interventions by identifying at-risk individuals across different academic phases.

-----

## Project Overview

This project applies **supervised learning techniques** to predict student dropout risk in international education programs. Using **XGBoost** and **Deep Neural Networks (DNN)**, the analysis aims to maximize **Recall for the "Dropout" class (Class 0)**, enabling timely interventions and data-driven decisions to improve retention and academic outcomes.

The project demonstrates a comprehensive machine learning pipeline, from data preparation through advanced model optimization and interpretation.

-----

## Data Source and Preprocessing

The dataset includes student demographics, engagement, and academic performance features, segmented across **three stages of the study journey**. Key features cover **age, gender, nationality, course information, attendance, and assessment results.**

> **Important Notice: Synthetic Data Only**
>
> This dataset is **100% synthetic** and was generated solely for demonstration purposes. It does not contain any real data.

### Key Preprocessing Steps

  * **Cleaning & Anomaly Detection:** Handled missing values, removed duplicates, and employed the **Isolation Forest** algorithm to identify and manage anomalies.
  * **Feature Engineering:** Derived new, powerful predictive features, including module completion counts and student **attendance rates**.
  * **Imbalance Handling:** Utilized **class weighting** across both models to effectively address the highly imbalanced nature of the dataset (many more completers than dropouts).
  * **Exploratory Data Analysis (EDA):** Performed initial visualization of distributions, feature relationships, and correlations relevant to dropout risk.

-----

## Approach: The Three-Stage Pipeline

The analysis follows a strict, progressive machine learning pipeline, focusing on model optimization across the academic timeline:

| Stage | Focus | Key Techniques |
| :--- | :--- | :--- |
| **Stage 1: Baseline** | Establish initial model performance metrics. | Default XGBoost and NN training. |
| **Stage 2: Mid-stage Optimization** | Tune hyperparameters to maximize **Recall** using mid-stage data features. | Advanced optimization with **Optuna** (XGBoost) and **Keras Tuner** (NN). |
| **Stage 3: Late-stage Evaluation** | Validate the best models on the final, most predictive feature set. | Final model retraining and comprehensive metric comparison. |


## File Structure

```
.
├── data/
│   └── synthetic_student_data_25000.csv
├── notebooks/
│   ├── Stage1_predicting_student_dropout_xgboost_nn.ipynb
│   ├── Stage2_predicting_student_dropout_xgboost_nn.ipynb
│   └── Stage3_predicting_student_dropout_xgboost_nn.ipynb
├── requirements/
│   ├── stage1_baseline.txt
│   ├── stage2_optimization.txt
│   └── stage3_evaluation.txt
└── src/
    ├── stage1_baseline_modelling/
    ├── stage2_model_optimization/
    └── stage_3_evaluation_comparison/
```

-----

##  How to run the code

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Student-Retention-Prediction-Using-XGBoost-and-Neural-Networks.git
    cd Student-Retention-Prediction-Using-XGBoost-and-Neural-Networks
    ```

2.  **Install dependencies:**

    ```bash
    # Install dependencies for the final stage, which covers all required libraries.
    pip install -r requirements/stage3_evaluation.txt
    ```

3.  **Run the notebooks:**
    Launch Jupyter and execute the notebooks in numerical order (`Stage1` $\rightarrow$ `Stage2` $\rightarrow$ `Stage3`) located in the `notebooks/` directory.

-----

Here is the dedicated **Skills & Employer Highlights** section, ready for your professional documents:

## Skills & Employer Highlights

This project demonstrates practical skills in:

* **Supervised Learning:** Building and evaluating robust models using **XGBoost** and **Deep Neural Networks**.
* **Model Optimization:** Advanced hyperparameter tuning with specialized frameworks **Optuna** and **Keras Tuner**.
* **Data Preparation & Engineering:** Feature Engineering, robust outlier detection using **Isolation Forest**, and handling imbalanced data (class weighting).
* **Model Evaluation:** Focused on business-critical metrics like **Recall (Class 0)** and **AUC-ROC** for intervention systems.
* **Interpretability & Visualization:** Used **PCA** and **t-SNE** for dimensionality reduction, visualization, and cluster analysis.
* **Python Ecosystem & Tools:** **pandas**, **NumPy**, **Scikit-learn**, **XGBoost**, **Optuna**, **TensorFlow/Keras**, **Keras Tuner**, **matplotlib**, **seaborn**.

**Impact:** This analysis extracts actionable insights from student data, identifies at-risk individuals across key academic stages, and supports **data-driven decision-making** to maximize student retention and programme success.
