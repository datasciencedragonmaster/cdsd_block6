#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Library Imports '''
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,  precision_recall_curve, auc, log_loss
from sklearn.utils import resample
import pickle

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import gower
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA, TruncatedSVD
from kmodes.kmodes import KModes

import streamlit as st
# import lib_cfasf_churn_data as data
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Separating features into types
cat_seniority = ['Entry-Level', 'Early/Mid-Level', 'Mid-Level', 'Senior-Level', 'Executive-Level', 'C-Suite', 'Other']
col_categorical_ordinal = ['seniority']
col_categorical_nominal = ['gender', 'type_cfai_membership', 'employment_status', 'employer_type'] 
col_numeric = ['age', 'is_in_france', 'year_joined', 'duration_membership_years', 'duration_program_years', 'is_on_professional_leave']
col_categorical = col_categorical_nominal + col_categorical_ordinal
categorical_features = ['gender', 'type_cfai_membership', 'employment_status', 'employer_type', 'seniority'] 
# Define the paths of saved models
path_model_lr_std = "./model_cfasf_membership_churn_lr.pkl" 
path_model_lr_best = "./model_cfasf_membership_churn_lr_best.pkl" 
path_model_rf_std = "./model_cfasf_membership_churn_rf.pkl" 
path_model_rf_best = "./model_cfasf_membership_churn_rf_best.pkl" 
# Define the hyperparameter grid for logistic regression
param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],   # Regularization strength, 0.001, 0.01, 
    'classifier__penalty': ['l1', 'l2'],               # Type of regularization
    'classifier__solver': ['liblinear', 'saga'],           # Solver compatible with L1 or L2, liblinear, lbfgs
    'classifier__class_weight': [None, 'balanced'],    # Handling class imbalance
    'classifier__max_iter': [1000, 5000, 10000, 20000]                    # Maximum iterations for convergence
}
# Define the parameter grid for RandomizedSearch
param_distributions_rf = {
    'classifier__n_estimators': [100, 200, 300, 400],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2'],  # Avoid 'None'
    'classifier__bootstrap': [True, False]
}
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Data Preprocessing '''
def get_preprocessor(data):
    # Define encoders for different features
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=[cat_seniority]), col_categorical_ordinal),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), col_categorical_nominal),  
            ('scaler', StandardScaler(), col_numeric)   # Removed StandardScaler for numeric features
        ],
        remainder='passthrough'  # Keep columns not specified in transformers
    )
    print(f"Number of selected columns is {len(col_categorical_nominal) + len(col_categorical_ordinal) + len(col_numeric)}")

    # Define features and target
    kept_columns = col_categorical_nominal + col_categorical_ordinal + col_numeric
    X = data[kept_columns]
    y = data['churned']  # Target variable
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return preprocessor, X_train, X_test, y_train, y_test
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Model Development '''
def get_pipeline(model, preprocessor):
    # Logistic Regression & Random Forest Pipelines
    if model == 'lr':
        pipeline_lr = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=10000, class_weight='balanced')) #LogisticRegression(random_state=42, class_weight='balanced')
        ])
        pipeline = pipeline_lr
    elif model == 'rf':
        pipeline_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced')) 
        ])
        pipeline = pipeline_rf
    return pipeline

def get_trained_model(pipeline, X_train, y_train, model_type, obj):
    # Training models
    print(f"Object Type: {type(pipeline)}")
    # if isinstance(pipeline, Pipeline):
    classifier = pipeline.named_steps['classifier']
    print(f"{classifier}")
    if model_type == 'standard':
        pipeline.fit(X_train, y_train)
        # print("Parameters found: ", classifier.get_params())
        return pipeline
    elif model_type == 'best':
        if isinstance(classifier, LogisticRegression):
            obj.write("***Running Grid Search ...***")
            grid_search_lr = GridSearchCV(estimator=pipeline, param_grid=param_grid_lr, 
                    cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1, verbose=3, return_train_score=True)
                            # Fit the grid search to your training data (X_train, y_train)
            grid_search_lr.fit(X_train, y_train)
            model_lr_best = grid_search_lr.best_estimator_
            params_lr_best = grid_search_lr.best_params_
            score_lr_best = grid_search_lr.best_score_
            cv_results_lr_best = grid_search_lr.cv_results_
            obj.write("**Grid Search done!**")
            return grid_search_lr
        elif isinstance(classifier, RandomForestClassifier):
            obj.write("**Running Random Search ...**")
            random_search_rf = RandomizedSearchCV(pipeline, param_distributions_rf, n_iter=50, scoring='accuracy', # 'f1', instead of scoring='accuracy'
                    n_jobs=-1, cv=5, random_state=42, verbose=2)
            random_search_rf.fit(X_train, y_train)
            model_rf_best = random_search_rf.best_estimator_
            params_rf_best = random_search_rf.best_params_
            score_rf_best = random_search_rf.best_score_
            cv_results_rf_best = random_search_rf.cv_results_
            obj.write("**Random Search done!**")
            return random_search_rf

# Predictions
def get_model_predictions(model, X_test):
    classifier = model.named_steps['classifier']  
    print(classifier.get_params()) 
    y_pred = model.predict(X_test)
    y_pred_churn_proba = model.predict_proba(X_test)[:, 1]
    y_pred_both_proba = model.predict_proba(X_test)  
    return y_pred, y_pred_churn_proba, y_pred_both_proba

# Evaluations
def evaluate_model(model, y_true, y_pred, y_pred_proba):
    classifier = model.named_steps['classifier']  
    params = classifier.get_params()
    print("Model Params:\t", params)
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:\t", acc)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print("AUC-ROC Score:\t", roc_auc)
    clf_report = classification_report(y_true, y_pred)
    print("Classification Report:\n", clf_report)
    cm = confusion_matrix(y_true, y_pred)
    print("Classification Report:\n", cm)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    print("\n")
    return params, acc, roc_auc, clf_report, cm

# Coefficients
def get_coefficients(model, nominal_features, ordinal_features, numeric_features):
    classifier = model.named_steps['classifier']  
    onehot_feature_names = model.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out(nominal_features)
    ordinal_feature_names = ordinal_features  # Use the names as they are for ordinal encoded features
    final_feature_names = list(ordinal_feature_names) + list(onehot_feature_names) + numeric_features
    feature_importance = pd.DataFrame([])
    if isinstance(classifier, LogisticRegression):
        print("The object is an instance of LogisticRegression.")
        coefficients = classifier.coef_.flatten()
        # coef =  model.named_steps['classifier'].coef_
        bias = classifier.intercept_.flatten()
        # Combine feature names with coefficients
        feature_importance = pd.DataFrame( {'Feature': final_feature_names, 'Coefficient': coefficients} )
        # Sort by the absolute value of the coefficients for better interpretability
        feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
        feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)
        # print(feature_importance)
        return coefficients, bias, feature_importance
    elif isinstance(classifier, RandomForestClassifier):
        print("The object is an instance of RandomForestClassifier.")
        feature_importances = classifier.feature_importances_.flatten()
        # print(feature_importances)
        feature_importance = pd.DataFrame( {'Feature': final_feature_names, 'Importance': feature_importances} )
        feature_importance['Abs_Coefficient'] = feature_importance['Importance'].abs()
        feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)
        # print(feature_importances)
        return feature_importances, 0, feature_importance

# Display metrics
def display_metrics(model, model_choice, obj, params, acc, roc_auc, report, cm, coef, bias, impt):
    # print(params)
    # print(pd.DataFrame(params, index=[0]))
    with obj:
        st.subheader("Metrics:", divider="gray")
        expander1 = st.expander("Examine metrics", expanded=False)
        with expander1:
            # expander = st.expander("Examine metrics")
            # obj.write(f"Params: {params}")
            # st.write("Params:", params)
            params_T = pd.DataFrame(params, index=[0])
            st.write(f"Hyperparameters:")
            st.dataframe(params_T)
            st.write(f"Accuracy: {acc}\n\nROC AUC: {roc_auc}")
            st.text(report)
            st.text(f"Confusion Matrix:\n {cm}")
            # obj.write(f"Coefficients: {coef}")
        #--------------------------------------------------------------------------------------------------------------------------------------------
        st.subheader("Coefficients:", divider="gray")
        expander2 = st.expander("Examine coefficients", expanded=False)
        with expander2:
            fig = go.Figure(go.Bar(
                x=impt['Abs_Coefficient'], y=impt['Feature'],
                orientation='h', marker=dict(color='skyblue') )) # Horizontal bar chart # Optional: set bar color
            # Update layout for the chart
            # print(f"HIIIII {model}")
            fig.update_layout(
                height=len(impt)*20, title=f'Feature Importances in {model} {model_choice}',
                xaxis_title='Importance', yaxis_title='Feature',
                yaxis={'autorange':'reversed'})  # Reverses the y-axis for highest importance at the top
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Coeffients:")
            st.dataframe(coef.reshape(1,-1))
            if bias != 0:
                st.write(f"Intercept:")
                st.dataframe(bias)

# Make predictions
def get_model(model, type):
    model_mapping = {('Logistic Regression', 'Standard Model'): path_model_lr_std, ('Logistic Regression', 'Best Model (Grid Search)'): path_model_lr_best, 
                    ('Random Forest', 'Standard Model'): path_model_rf_std, ('Random Forest', 'Best Model (Random Search)'): path_model_rf_best,}
    model_pkl_filepath = model_mapping.get((model, type), "Path not found")
    print(f'Current fetched model filepath: {model_pkl_filepath}')
    model = load_model(model_pkl_filepath)
    # print(f'Current loaded model: {model}')   
    return model

# Make predictions
def get_prediction(model, sample):
    # print(f'Current loaded model:  {model}')   
    print(f'Current loaded model type:  {type(model)}')   
    print(f'Current sample: {sample}')  
    # y_pred_specific = model.predict(X_test[0:1])
    y_pred_sample = model.predict(sample)
    y_pred_proba_sample = model.predict_proba(sample)  # For AUC-ROC
    return y_pred_sample, y_pred_proba_sample

#------------------------------------------------------------------------------------------------------------------------------------------------------
def save_model(model_pkl_file, model):
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)    

def load_model(model_pkl_file):
    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(file)
    return model
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Model Stats Display '''
def display_model_stats(model, df, obj):
    preprocessor, X_train, X_test, y_train, y_test = get_preprocessor(df)
    if model == 'Logistic Regression':
        obj.subheader("Logistic Regression Model", anchor='lr')
        obj.write("Choose Model Type")
        radio_cols = obj.columns(2)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        model_choice = radio_cols[0].radio(" ", options=['Standard Model', 'Best Model (Grid Search)'], index=0, key="key_model_lr")
        
        pipeline_lr = get_pipeline('lr', preprocessor)
        if model_choice == 'Standard Model':
            model_pkl_filepath = path_model_lr_std
            if not Path(model_pkl_filepath).exists():
                model_lr = get_trained_model(pipeline_lr, X_train, y_train, 'standard', obj)
                save_model(model_pkl_filepath, model_lr)
            else:
                model_lr = load_model(model_pkl_filepath)
            y_pred_lr, y_pred_lr_churn_proba, y_pred_lr_both_proba = get_model_predictions(model_lr, X_test)
            params_lr, acc_lr, roc_auc_lr, report_lr, cm_lr = evaluate_model(model_lr, y_test, y_pred_lr, y_pred_lr_churn_proba)
            coef_lr, bias_lr, imptf_lr = get_coefficients(model_lr, col_categorical_nominal, col_categorical_ordinal, col_numeric)    
            # Display model metrics
            display_metrics(model, model_choice, obj, params_lr, acc_lr, roc_auc_lr, report_lr, cm_lr, coef_lr, bias_lr, imptf_lr) 
        elif model_choice == 'Best Model (Grid Search)':
            model_pkl_filepath = path_model_lr_best
            if not Path(model_pkl_filepath).exists():
                gridsearch_lr = get_trained_model(pipeline_lr, X_train, y_train, 'best', obj)
                save_model(model_pkl_filepath, gridsearch_lr)
            else:
                gridsearch_lr = load_model(model_pkl_filepath)
            model_lr_best = gridsearch_lr.best_estimator_
            y_pred_lr_best, y_pred_lr_best_churn_proba, y_pred_lr_best_both_proba = get_model_predictions(model_lr_best, X_test)
            params_lr_best, acc_lr_best, roc_auc_lr_best, report_lr_best, cm_lr_best = evaluate_model(model_lr_best, y_test, y_pred_lr_best, y_pred_lr_best_churn_proba)
            coef_lr_best, bias_lr_best, imptf_lr_best = get_coefficients(model_lr_best, col_categorical_nominal, col_categorical_ordinal, col_numeric)                  
            # Display model metrics
            display_metrics(model, model_choice, obj, params_lr_best, acc_lr_best, roc_auc_lr_best, report_lr_best, cm_lr_best, coef_lr_best, bias_lr_best, imptf_lr_best)
             
        # Prediction Tool for Selected Model
        obj.subheader("Predict Churn", divider='grey', anchor='predict_lr')
        if obj.checkbox("Enable Prediction with Selected Model", key='chkbox_predict_lr'):
            expander_predict_lr = obj.expander("Enter values for prediction:", expanded=True)
            with expander_predict_lr:
                model_features = list(df.columns.drop(['person_id', 'mailing_locality', 'year_last_membership', 'is_society_member', 'code_group_membership_change','churned']))
                # print(mdl.col_categorical_nominal + mdl.col_categorical_ordinal + mdl.col_numeric)
                print(model_features)
                current_year = datetime.now().year
                sel_gender_lr = st.selectbox("Gender", options=df['gender'].unique(), key='sel_gender_lr')
                sel_age_lr = st.slider("Age", min_value=20, max_value=100, value=40, key='sel_age_lr') # number_input
                sel_is_in_france_lr = st.selectbox("In France?", options=df['is_in_france'].unique(), key='sel_is_in_france_lr')
                sel_year_joined_lr = st.slider("Year Joined", min_value=2000, max_value=current_year+1, value=current_year-4, key='sel_year_joined_lr')
                sel_type_cfai_membership_lr = st.selectbox("Type CFAI membership", options=df['type_cfai_membership'].unique(), key='sel_type_cfai_membership_lr')
                sel_duration_program_years_lr = st.slider("Duration in CFA Program (Yrs)", min_value=2, max_value=datetime.now().year+1-2000, value=3, key='sel_duration_program_years_lr')
                sel_duration_membership_years_lr = st.slider("Duration of Membership (Yrs)", min_value=1, max_value=50, value=6, key='sel_duration_membership_years_lr')
                sel_is_on_professional_leave_lr = st.selectbox("On Professional Leave?", options=df['is_on_professional_leave'].unique(), key='sel_is_on_professional_leave_lr')
                sel_employment_status_lr = st.selectbox("Employment Status", options=df['employment_status'].unique(), key='sel_employment_status_lr')
                sel_employer_type_lr = st.selectbox("Employment Type", options=sorted(df['employer_type'].unique()), index=1, key='sel_employer_type_lr')
                sel_seniority_lr = st.selectbox("Seniority", options=cat_seniority, index=3, key='sel_seniority_lr')

            # Prepare input data for prediction
            sample_member = pd.DataFrame({
                'gender': [sel_gender_lr], 'age': [sel_age_lr], 'is_in_france': [sel_is_in_france_lr], 
                'year_joined': [sel_year_joined_lr], 'type_cfai_membership': [sel_type_cfai_membership_lr], 
                'duration_program_years': [sel_duration_program_years_lr], 'duration_membership_years': [sel_duration_membership_years_lr], 
                'is_on_professional_leave': [sel_is_on_professional_leave_lr], 'employment_status': [sel_employment_status_lr], 
                'employer_type': [sel_employer_type_lr], 'seniority': [sel_seniority_lr], })
                
            loaded_model = get_model(model, model_choice)
            prediction, proba = get_prediction(loaded_model, sample_member)
            print(model, model_choice)
            print(prediction, proba)
            obj.subheader("Prediction Results", divider='grey', anchor='result')
            prediction_result = f'''**Prediction:** 
                        \nUsing the **:blue[{model} {model_choice}]**, 
                        \nthe above sample member will :red[{'CHURN' if prediction[0] == 1 else 'NOT CHURN'}] 
                        with a probability of {round(proba.flatten()[1]*100, 2) if prediction[0] == 1 else round(proba.flatten()[0]*100,2)}%.'''
            obj.markdown(prediction_result)

                
    elif model == 'Random Forest':
        obj.subheader("Random Forest Model", anchor='rf')
        obj.write("Choose Model Type")
        radio_cols = obj.columns(2)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        model_choice = radio_cols[0].radio(" ", options=['Standard Model', 'Best Model (Random Search)'], index=0, key="key_model_rf")
       
        pipeline_rf = get_pipeline('rf', preprocessor)
        if model_choice == 'Standard Model':
            model_pkl_filepath = path_model_rf_std
            if not Path(model_pkl_filepath).exists():
                model_rf = get_trained_model(pipeline_rf, X_train, y_train, 'standard', obj)
                save_model(model_pkl_filepath, model_rf)
            else:
                model_rf = load_model(model_pkl_filepath)
            y_pred_rf, y_pred_rf_churn_proba, y_pred_rf_both_proba = get_model_predictions(model_rf, X_test)
            params_rf, acc_rf, roc_auc_rf, report_rf, cm_rf = evaluate_model(model_rf, y_test, y_pred_rf, y_pred_rf_churn_proba)
            ftr_rf, bias_rf, imptf_rf = get_coefficients(model_rf, col_categorical_nominal, col_categorical_ordinal, col_numeric)    
            # Display model metrics
            display_metrics(model, model_choice, obj, params_rf, acc_rf, roc_auc_rf, report_rf, cm_rf, ftr_rf, bias_rf, imptf_rf) 
        elif model_choice == 'Best Model (Random Search)':
            model_pkl_filepath = path_model_rf_best
            if not Path(model_pkl_filepath).exists():
                randomsearch_rf = get_trained_model(pipeline_rf, X_train, y_train, 'best', obj)
                save_model(model_pkl_filepath, randomsearch_rf)
            else:
                randomsearch_rf = load_model(model_pkl_filepath)
            model_rf_best = randomsearch_rf.best_estimator_
            y_pred_rf_best, y_pred_rf_best_churn_proba, y_pred_rf_best_both_proba = get_model_predictions(model_rf_best, X_test)
            params_rf_best, acc_rf_best, roc_auc_rf_best, report_rf_best, cm_rf_best = evaluate_model(model_rf_best, y_test, y_pred_rf_best, y_pred_rf_best_churn_proba)
            coef_rf_best, bias_rf_best, imptf_rf_best = get_coefficients(model_rf_best, col_categorical_nominal, col_categorical_ordinal, col_numeric)                  
            # Display model metrics
            display_metrics(model, model_choice, obj, params_rf_best, acc_rf_best, roc_auc_rf_best, report_rf_best, cm_rf_best, coef_rf_best, bias_rf_best, imptf_rf_best)
    
        # Prediction Tool for Selected Model
        obj.subheader("Predict Churn", divider='grey', anchor='predict_rf')
        if obj.checkbox("Enable Prediction with Selected Model", key='chkbox_predict_rf'):
            expander_predict_rf = obj.expander("Enter values for prediction:", expanded=True)
            with expander_predict_rf:
                model_features = list(df.columns.drop(['person_id', 'mailing_locality', 'year_last_membership', 'is_society_member', 'code_group_membership_change','churned']))
                # print(mdl.col_categorical_nominal + mdl.col_categorical_ordinal + mdl.col_numeric)
                print(model_features)
                current_year = datetime.now().year
                sel_gender_rf = st.selectbox("Gender", options=df['gender'].unique())
                sel_age_rf = st.slider("Age", min_value=20, max_value=100, value=40)    # number_input
                sel_is_in_france_rf = st.selectbox("In France?", options=df['is_in_france'].unique())
                sel_year_joined_rf = st.slider("Year Joined", min_value=2000, max_value=current_year+1, value=current_year-4)
                sel_type_cfai_membership_rf = st.selectbox("Type CFAI membership", options=df['type_cfai_membership'].unique())
                sel_duration_program_years_rf = st.slider("Duration in CFA Program (Yrs)", min_value=2, max_value=datetime.now().year+1-2000, value=3)
                sel_duration_membership_years_rf = st.slider("Duration of Membership (Yrs)", min_value=1, max_value=50, value=6)
                sel_is_on_professional_leave_rf = st.selectbox("On Professional Leave?", options=df['is_on_professional_leave'].unique())
                sel_employment_status_rf = st.selectbox("Employment Status", options=df['employment_status'].unique())
                sel_employer_type_rf = st.selectbox("Employment Type", options=sorted(df['employer_type'].unique()), index=1)
                sel_seniority_rf = st.selectbox("Seniority", options=cat_seniority, index=3)

            # Prepare input data for prediction
            sample_member = pd.DataFrame({
                'gender': [sel_gender_rf], 'age': [sel_age_rf], 'is_in_france': [sel_is_in_france_rf], 
                'year_joined': [sel_year_joined_rf], 'type_cfai_membership': [sel_type_cfai_membership_rf], 
                'duration_program_years': [sel_duration_program_years_rf], 'duration_membership_years': [sel_duration_membership_years_rf], 
                'is_on_professional_leave': [sel_is_on_professional_leave_rf], 'employment_status': [sel_employment_status_rf], 
                'employer_type': [sel_employer_type_rf], 'seniority': [sel_seniority_rf], })
                
            loaded_model = get_model(model, model_choice)
            prediction, proba = get_prediction(loaded_model, sample_member)
            print(model, model_choice)
            print(prediction, proba)
            obj.subheader("Prediction Results", divider='grey', anchor='result')
            prediction_result = f'''**Prediction:** 
                        \nUsing the **:blue[{model} {model_choice}]**, 
                        \nthe above sample member will :red[{'CHURN' if prediction[0] == 1 else 'NOT CHURN'}] 
                        with a probability of {round(proba.flatten()[1]*100, 2) if prediction[0] == 1 else round(proba.flatten()[0]*100,2)}%.'''
            obj.markdown(prediction_result)
    
    
    
    
    return model_choice    
        

       
    
