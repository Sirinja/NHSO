# NHSO
# IMC Better Prediction Model

## Project Goal

The goal of this project is to build a predictive model to identify patients who are likely to show improvement in their ADL (Activities of Daily Living) score after receiving care, specifically predicting the `imc_better` target variable (1 if ADL improves, 0 otherwise).

## Data

The analysis is based on the dataset `20250903_x_ imc.csv`. Key features used in the model include:

*   Patient demographics (e.g., age, sex)
*   Disease information (`disease_id`)
*   Charlson Comorbidity Index (CCI) derived from ICD-10 codes
*   Length of Stay (LOS)
*   Resource Intensity per Day (`rw_los_ratio`, `rw_nhso`, `adjrw_nhso`)
*   Waiting days (`waitingdays`, `waiting_range`)

## Methodology

1.  **Data Preprocessing:**
    *   Handling missing values (imputation).
    *   Encoding categorical features (e.g., `age_group`).
    *   Calculating derived features (CCI, LOS, `rw_los_ratio`, `adl_change`, `imc_better`, `waiting_range`).
2.  **Feature Selection:**
    *   Identified important features based on feature importances from an initial model.
3.  **Model Training:**
    *   Evaluated multiple classification models (Logistic Regression, Random Forest, HistGradientBoosting).
    *   Selected Random Forest as the best initial model.
    *   Fine-tuned the Random Forest model using GridSearchCV, optimizing for recall.
4.  **Evaluation:**
    *   Evaluated the fine-tuned model on a held-out test set using metrics such as ROC AUC, PR AUC, precision, recall, and F1-score at different thresholds (Youden's J and a specific recall band).
5.  **Model Saving:**
    *   Saved the final model, including the preprocessing steps, as a pipeline using `joblib`.

## Results

The models developed achieved performance metrics (AUCs around 0.55-0.60) that are only slightly better than random chance. This indicates that predicting `imc_better` with the current features is a challenging task. While fine-tuning improved recall for the positive class (ability to identify patients who will improve), it did not significantly boost overall predictive power.

Key performance metrics on the test set for the fine-tuned model:

*   **ROC AUC:** ~0.59
*   **PR AUC:** ~0.56
*   **Performance at Youden's J threshold (~0.524):**
    *   Recall: ~0.583
    *   Precision: ~0.563
    *   F1-score: ~0.573
*   **Performance at Recall band [0.70-0.80] threshold (~0.497):**
    *   Recall: ~0.717
    *   Precision: ~0.546
    *   F1-score: ~0.620

## Usage

The trained model, including the necessary preprocessing steps, is saved as a pipeline in the file `model_with_preprocessing_pipeline.joblib`.

You can load this pipeline and use it to make predictions on new data:
