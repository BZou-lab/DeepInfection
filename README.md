# DeepInfection

This repository contains the R scripts and Jupyter notebooks used for the study: **"An Interpretable Deep Learning Framework for Accurate and Robust Postoperative Infection Risk Prediction of Intensive Care Unit Patients."** The scripts implement and evaluate deep learning and machine learning models, including interpretable PermFIT models, to predict the risk of postoperative infection in ICU patients. 

## Repository Contents

The repository includes two Jupyter notebooks for data preparation and two main R scripts:
1. `Raw_data_extraction.ipynb` - Extracts and pre-processes raw ICU patient data from the original source.
2. `Clean_data.ipynb` - Cleans and transforms the raw data into the final format used for modeling.
3. `standard models.R` - Implements standard predictive models: DNN, RF, SVM, and XGBoost.
4. `PermFIT models.R` - Implements interpretable PermFIT models based on the four standard models: PermFIT-DNN, PermFIT-RF, PermFIT-SVM, and PermFIT-XGBoost.

### 1. `Raw_data_extraction.ipynb`
  Extracts and merges raw MIMIC-III data to identify adult ICU patients who underwent surgery and flags postoperative infections.
  
### 2. `Clean_data.ipynb`
  Cleans the extracted dataset, handles missing values, performs feature engineering, and outputs a modeling-ready file.
  
### 3. `standard models.R`
This script provides a baseline comparison of predictive models, including:
- **DNN**: A deep neural network model for flexible learning of complex patterns in data.
- **RF**: A random forest model that averages multiple decision trees to improve predictive accuracy and control overfitting.
- **SVM**: A support vector machine model that classifies data by finding the optimal hyperplane.
- **XGBoost**: An XGBoost model, which is an optimized gradient-boosting algorithm designed for improved speed and performance.

The `standard models.R` script:
- Accepts model selection and cross-validation fold parameters via command-line arguments.
- Loads the ICU patient dataset and applies cross-validation to each model.
- Evaluates model performance using metrics such as accuracy, AUC, and PRAUC, and saves the results to CSV files.

### 4. `PermFIT models.R`
This script extends the standard models with the PermFIT framework to produce interpretable risk predictions. The four PermFIT-based models are:
- **PermFIT-DNN**: An interpretable version of the deep neural network using permutation feature importance.
- **PermFIT-RF**: An interpretable random forest model with feature importance measures.
- **PermFIT-SVM**: An interpretable SVM model, leveraging permutation importance to rank feature contributions.
- **PermFIT-XGB**: An interpretable XGBoost model that includes feature importance evaluation.

The `PermFIT models.R` script:
- Uses the PermFIT approach to identify and report feature importance alongside risk predictions.
- Loads the ICU patient dataset and performs cross-validation for each PermFIT model.
- Evaluates and saves both feature importance scores and prediction metrics, offering insight into which features contribute to infection risk prediction.
