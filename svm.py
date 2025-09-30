#!/usr/bin/env python
# coding: utf-8

"""
Pima Indians Diabetes Dataset - SVM Classification Pipeline
Main pipeline script with minimal changes from the original notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def main():
    """Main pipeline function"""
    
    # ========== STEP 1: Load Data ==========
    print("="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    # Update this path to your data location
    df = pd.read_csv('D:\Private\Python\data\datasets\pima_indians_diabetes_with_header.csv')
    print(f"Data loaded successfully: {df.shape}")
    
    # ========== STEP 2: Initial Data Visualization ==========
    print("\n" + "="*60)
    print("STEP 2: INITIAL DATA EXPLORATION")
    print("="*60)
    
    # Create correlation matrix
    correlation_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,  # Show correlation values
                fmt='.2f',   # Format to 2 decimal places
                cmap='coolwarm',  # Color scheme
                center=0,    # Center colormap at 0
                square=True, # Make cells square
                linewidths=1,  # Add gridlines
                cbar_kws={"shrink": 0.8})  # Adjust colorbar size
    
    plt.title('Correlation Heatmap - Pima Indians Diabetes Dataset (Original)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # ========== STEP 3: Data Overview ==========
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Statistics:")
    print(df.describe())
    
    print("\nNull values check:")
    print(df.isnull().sum())
    
    cols = list(df.columns)
    print("\nColumn list:")
    for i, c in enumerate(cols, start=1):
        print(f"{i}. {c}")
    
    print("\nDataset Info:")
    print(df.info())
    
    zero_counts = (df == 0).sum()
    print("\nZero counts in each column:")
    print(zero_counts)
    
    # ========== STEP 4: Age Group Analysis ==========
    print("\n" + "="*60)
    print("STEP 4: AGE GROUP ANALYSIS")
    print("="*60)
    
    # Filter rows where Age >= 40
    age_over_40 = df[df['Age'] >= 40]
    print(f"Number of rows with Age >= 40: {len(age_over_40)}")
    print("\nAge >= 40 Statistics:")
    print(age_over_40.describe())
    
    # Filter rows where Age < 40
    age_below_40 = df[df['Age'] < 40]
    print(f"\nNumber of rows with Age < 40: {len(age_below_40)}")
    print("\nAge < 40 Statistics:")
    print(age_below_40.describe())
    
    # ========== STEP 5: Data Imputation - Glucose and BloodPressure ==========
    print("\n" + "="*60)
    print("STEP 5: DATA IMPUTATION - GLUCOSE AND BLOODPRESSURE")
    print("="*60)
    
    # Create a copy to preserve original data
    df_filled = df.copy()
    
    # Define age groups
    age_over_or_equal_40_mask = df_filled['Age'] >= 40
    age_40_under_mask = df_filled['Age'] < 40
    
    # Calculate statistics for each age group (excluding zeros)
    # For Age >= 40
    glucose_median_over_40 = df_filled.loc[age_over_or_equal_40_mask & (df_filled['Glucose'] != 0), 'Glucose'].median()
    bp_mean_over_40 = df_filled.loc[age_over_or_equal_40_mask & (df_filled['BloodPressure'] != 0), 'BloodPressure'].mean()
    
    # For Age < 40
    glucose_median_40_or_under = df_filled.loc[age_40_under_mask & (df_filled['Glucose'] != 0), 'Glucose'].median()
    bp_mean_40_or_under = df_filled.loc[age_40_under_mask & (df_filled['BloodPressure'] != 0), 'BloodPressure'].mean()
    
    # Fill zeros for Age >= 40
    df_filled.loc[(age_over_or_equal_40_mask) & (df_filled['Glucose'] == 0), 'Glucose'] = glucose_median_over_40
    df_filled.loc[(age_over_or_equal_40_mask) & (df_filled['BloodPressure'] == 0), 'BloodPressure'] = bp_mean_over_40
    
    # Fill zeros for Age < 40
    df_filled.loc[(age_40_under_mask) & (df_filled['Glucose'] == 0), 'Glucose'] = glucose_median_40_or_under
    df_filled.loc[(age_40_under_mask) & (df_filled['BloodPressure'] == 0), 'BloodPressure'] = bp_mean_40_or_under
    
    # Display the statistics used for imputation
    print("=== Imputation Values ===")
    print(f"Age >= 40:")
    print(f"  Glucose median: {glucose_median_over_40:.2f}")
    print(f"  BloodPressure mean: {bp_mean_over_40:.2f}")
    print(f"\nAge < 40:")
    print(f"  Glucose median: {glucose_median_40_or_under:.2f}")
    print(f"  BloodPressure mean: {bp_mean_40_or_under:.2f}")
    
    # Check how many values were filled
    print("\n=== Number of Zeros Filled ===")
    print(f"Glucose zeros before: {(df['Glucose'] == 0).sum()}")
    print(f"Glucose zeros after: {(df_filled['Glucose'] == 0).sum()}")
    print(f"BloodPressure zeros before: {(df['BloodPressure'] == 0).sum()}")
    print(f"BloodPressure zeros after: {(df_filled['BloodPressure'] == 0).sum()}")
    
    print("\nDataset after first imputation:")
    print(df_filled.describe())
    
    zero_counts = (df_filled == 0).sum()
    print("\nRemaining zeros after first imputation:")
    print(zero_counts)
    
    # ========== STEP 6: BMI Imputation ==========
    print("\n" + "="*60)
    print("STEP 6: BMI IMPUTATION")
    print("="*60)
    
    BMI_zero = df_filled[df_filled['BMI'] == 0]
    print(f"Rows with BMI = 0: {len(BMI_zero)}")
    print(BMI_zero)
    
    # Calculate median BMI (excluding zeros)
    bmi_median = df_filled[df_filled['BMI'] != 0]['BMI'].median()
    
    # Fill zeros with median
    df_filled.loc[df_filled['BMI'] == 0, 'BMI'] = bmi_median
    print(f"\nBMI filled with median: {bmi_median:.2f}")
    
    zero_counts = (df_filled == 0).sum()
    print("\nZero counts after BMI imputation:")
    print(zero_counts)
    
    # ========== STEP 7: Outcome Analysis ==========
    print("\n" + "="*60)
    print("STEP 7: OUTCOME ANALYSIS")
    print("="*60)
    
    # Filter by Outcome
    no_diabetes = df_filled[df_filled['Outcome'] == 0]
    has_diabetes = df_filled[df_filled['Outcome'] == 1]
    
    print("No Diabetes (Outcome=0) Statistics:")
    print(no_diabetes.describe())
    
    print("\nDiabetes (Outcome=1) Statistics:")
    print(has_diabetes.describe())
    
    # ========== STEP 8: Feature Selection - Drop Columns ==========
    print("\n" + "="*60)
    print("STEP 8: FEATURE SELECTION - DROPPING COLUMNS")
    print("="*60)
    
    # Drop the specified columns from the features
    df_filled = df_filled.drop(columns=['Insulin', 'Pregnancies'])
    print("Dropped columns: Insulin, Pregnancies")
    print("Columns remaining in features:", df_filled.columns.tolist())
    
    # Create correlation matrix after dropping columns
    correlation_matrix = df_filled.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap - After Dropping Insulin & Pregnancies', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    zero_counts = (df_filled == 0).sum()
    print("\nZero counts after dropping columns:")
    print(zero_counts)
    
    # ========== STEP 9: SkinThickness Imputation ==========
    print("\n" + "="*60)
    print("STEP 9: SKINTHICKNESS IMPUTATION BY OUTCOME")
    print("="*60)
    
    # Calculate mean SkinThickness for each outcome group (excluding zeros)
    mean_skin_outcome_0 = df_filled.loc[(df_filled['Outcome'] == 0) & (df_filled['SkinThickness'] != 0), 'SkinThickness'].mean()
    mean_skin_outcome_1 = df_filled.loc[(df_filled['Outcome'] == 1) & (df_filled['SkinThickness'] != 0), 'SkinThickness'].mean()
    
    # Fill zeros based on outcome
    df_filled.loc[(df_filled['Outcome'] == 0) & (df_filled['SkinThickness'] == 0), 'SkinThickness'] = mean_skin_outcome_0
    df_filled.loc[(df_filled['Outcome'] == 1) & (df_filled['SkinThickness'] == 0), 'SkinThickness'] = mean_skin_outcome_1
    
    # Display results
    print("=== Imputation Values ===")
    print(f"Mean SkinThickness for Outcome 0 (No Diabetes): {mean_skin_outcome_0:.2f}")
    print(f"Mean SkinThickness for Outcome 1 (Diabetes): {mean_skin_outcome_1:.2f}")
    
    print("\n=== Zeros Filled ===")
    print(f"SkinThickness zeros before: {(df['SkinThickness'] == 0).sum()}")
    print(f"SkinThickness zeros after: {(df_filled['SkinThickness'] == 0).sum()}")
    
    # Create correlation matrix after SkinThickness imputation
    correlation_matrix = df_filled.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap - After SkinThickness Imputation', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\nDataset after all imputations:")
    print(df_filled)
    
    # ========== STEP 10: Additional Feature Reduction ==========
    print("\n" + "="*60)
    print("STEP 10: ADDITIONAL FEATURE REDUCTION")
    print("="*60)
    
    # Drop additional columns
    df_filled = df_filled.drop(columns=['BloodPressure','DiabetesPedigreeFunction','Age'])
    print("Dropped columns: BloodPressure, DiabetesPedigreeFunction, Age")
    print("Columns remaining in features:", df_filled.columns.tolist())
    
    # Final correlation matrix
    correlation_matrix = df_filled.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap - Final Feature Set', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # ========== STEP 11: Train-Test Split ==========
    print("\n" + "="*60)
    print("STEP 11: TRAIN-TEST SPLIT")
    print("="*60)
    
    # Separate features and target
    X = df_filled.drop('Outcome', axis=1)
    y = df_filled['Outcome']
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.1,          # 10% for test
        stratify=y,             # Stratify based on Outcome
        random_state=42         # For reproducibility
    )
    
    # Verify the split
    print("=== Dataset Split Summary ===")
    print(f"Total samples: {len(df_filled)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(df_filled)*100:.1f}%)")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(df_filled)*100:.1f}%)")
    
    # Check outcome distribution
    print("\n=== Outcome Distribution ===")
    print("Original dataset:")
    print(df_filled['Outcome'].value_counts(normalize=True).sort_index())
    print(f"  No Diabetes (0): {(df_filled['Outcome']==0).sum()} samples")
    print(f"  Diabetes (1): {(df_filled['Outcome']==1).sum()} samples")
    
    print("\nTraining set:")
    print(y_train.value_counts(normalize=True).sort_index())
    print(f"  No Diabetes (0): {(y_train==0).sum()} samples")
    print(f"  Diabetes (1): {(y_train==1).sum()} samples")
    
    print("\nTest set:")
    print(y_test.value_counts(normalize=True).sort_index())
    print(f"  No Diabetes (0): {(y_test==0).sum()} samples")
    print(f"  Diabetes (1): {(y_test==1).sum()} samples")
    
    # ========== STEP 12: Feature Scaling ==========
    print("\n" + "="*60)
    print("STEP 12: FEATURE SCALING")
    print("="*60)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform both the training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    print(f"Scaled training set shape: {X_train_scaled.shape}")
    print(f"Scaled test set shape: {X_test_scaled.shape}")
    
    # ========== STEP 13: SVM Model Training ==========
    print("\n" + "="*60)
    print("STEP 13: SVM MODEL TRAINING")
    print("="*60)
    
    # Initialize the SVM classifier with RBF kernel
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    
    # Train the model
    print("Training the SVM Model...")
    print(f"Kernel: RBF")
    print(f"C parameter: 1.0")
    svm_model.fit(X_train_scaled, y_train)
    print("Model training complete!")
    
    # Make predictions
    y_pred = svm_model.predict(X_test_scaled)
    
    # ========== STEP 14: Model Evaluation ==========
    print("\n" + "="*60)
    print("STEP 14: MODEL EVALUATION")
    print("="*60)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visual representation of the confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes (0)', 'Diabetes (1)'], 
                yticklabels=['No Diabetes (0)', 'Diabetes (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=['No Diabetes (0)', 'Diabetes (1)'])
    print(report)
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    
    print("\n--- Additional Metrics ---")
    print(f"Specificity: {specificity:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    
    # ========== Pipeline Complete ==========
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal Model Performance:")
    print(f"  - Accuracy: {accuracy:.2%}")
    print(f"  - Specificity: {specificity:.2%}")
    print(f"  - Sensitivity: {sensitivity:.2%}")
    
    return svm_model, scaler, accuracy


if __name__ == "__main__":
    # Run the main pipeline
    model, scaler, accuracy = main()
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("Model and scaler are ready for deployment.")
    print("="*60)