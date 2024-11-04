#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Library Imports '''
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from datetime import datetime

import streamlit as st
# import extra_streamlit_components as stx
import lib_cfasf_churn_viz as viz
import lib_cfasf_churn_models as mdl
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Data Imports & Loads'''
path_dataset_kept = './datasets/2024-10-21 - data_cfasf_membership_kept_col.csv'
path_dataset_all = './datasets/2024-10-21 - data_cfasf_membership_all_col.csv'
def fetch_data(url):
    # Fetch data from URL here, and then clean it up.
    data = pd.read_csv(url, encoding='latin-1')
    return data
data_viz = fetch_data(path_dataset_all)
df_eda = data_viz.copy()        # for eda data viz
df_churn = data_viz.copy()      # for churn feature analysis
data_imported = fetch_data(path_dataset_kept)
df_model = data_imported.copy()
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Streamlit app '''
# Page Configuration
st.set_page_config(
    page_title="Churn Dashboard",  # Sets the browser tab title
    page_icon="📊",               # Sets the favicon (supports emojis, paths, or URLs)
    layout="wide",             # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",  # Options: "auto", "expanded", "collapsed"
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': 'https://www.example.com/bug',
        'About': "# This is a Streamlit app for churn analysis"
    }
)
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Top Section: Project Introduction
st.title("CFA Society France Membership Churn Analysis and Prediction", anchor='project')
multi = """
    This project aims to understand, interpret, and predict `membership churn` 
    with descriptive and predictive analytics.
    
    Anonymised membership data (as of Aug 5, 2024) is used with a total of 2331 elements (post removal of 4 invalid elements).\n
    The following dashboard has two objectives :
    1) provide insights which lead to better understanding and interpretation of membership churn
    2) determine and predict membership churn with an acceptable level of accuracy
    
    Thus helping stakeholders make better informed data-driven decisions on membership retention.
"""
st.markdown(multi)
# Section Break
st.markdown("---")
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Middle Section: Descriptive Analytics
st.header("Descriptive Analytics", anchor='descriptive')
# Row 1: Filtered Membership Data with Checkbox Filters (Horizontal Layout)
st.subheader("Exploratory Data Analysis of Membership Data", anchor='eda')
col1, col2 = st.columns([1, 9])  # Narrow col1 for the checkbox filters

with col1:
    for i in range(0):
        st.write("") # if we need to move wording down
    # Dropdown for feature selection
    viz_feature = st.selectbox(
        "Select feature to display breakdown and figure",
        options = viz.get_eda_features()
    )
    
with col2:
    # Horizontal checkboxes for membership filter
    st.write("Select Membership Status")
    checkbox_cols = st.columns(3)
    opt_all = checkbox_cols[0].checkbox("Full Population", value=True)
    opt_member = checkbox_cols[1].checkbox("Society Member", value=False)
    opt_non_member = checkbox_cols[2].checkbox("Not Society Member", value=False)
       
    # Filter data based on selected checkboxes 
    if opt_all or opt_member or opt_non_member:
        if opt_all:
            data_filtered = viz.fetch_eda_breakdown('all', viz_feature, df_eda, col1)
            figs_all = viz.display_eda_figure('all', viz_feature, df_eda, checkbox_cols[0])
        if opt_member:
            data_filtered = viz.fetch_eda_breakdown('member', viz_feature, df_eda, col1)
            figs_member = viz.display_eda_figure('member', viz_feature, df_eda, checkbox_cols[1])
        if opt_non_member:
            data_filtered = viz.fetch_eda_breakdown('non-member', viz_feature, df_eda, col1)
            figs_non_member = viz.display_eda_figure('non-member', viz_feature, df_eda, checkbox_cols[2])
    else:
        data_filtered = viz.fetch_eda_breakdown('none', viz_feature, df_eda, col1)   
        # figs_none = viz.display_eda_figure('none', viz_feature, df_viz)

# Section Break
st.markdown("---")
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Row 2: Feature Impact on Churn with Tabs
st.subheader("Feature Impact Analysis on Churn", anchor='feature')
lst_feature_tabs = viz.get_churn_features()
feature_tabs = st.tabs(lst_feature_tabs)

for i in range(len(lst_feature_tabs)):
    viz.display_churn_figure(i, df_churn, feature_tabs[i])

# Section Break
st.markdown("---")
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Bottom Section: Predictive Analytics
st.header("Predictive Analytics", anchor='predictive')
# Tabs for Model Selection (Logistic Regression, Random Forest)
lst_model_tabs = ["Logistic Regression", "Random Forest"]
# # Initialize session state to store the active tab index
# if "active_tab" not in st.session_state: st.session_state.active_tab = 0  # Default to the first tab
model_tabs = st.tabs(lst_model_tabs)
# chosen_id = stx.tab_bar(data=[
#     stx.TabBarItemData(id="tab1", title="✍️ To Do", description="Tasks to take care of"),
#     stx.TabBarItemData(id="tab2", title="📣 Done", description="Tasks taken care of"),])

for model, i in zip( lst_model_tabs, range(len(lst_model_tabs)) ):
    model_choice = mdl.display_model_stats(model, df_model, model_tabs[i])  
    
#------------------------------------------------------------------------------------------------------------------------------------------------------
### Side bar 
# st.sidebar.title("Navigation")
st.sidebar.header("Navigation Pane", anchor = '')
st.sidebar.markdown("""
    * [Project](#project)
    * [Descriptive Analytics](#descriptive)
        * [EDA](#eda)
        * [Feature Impact](#feature)
    * [Predictive Analytics](#predictive)   
        * [Models](#lr)
        * [Predict Churn](#predict_lr)
    * [Reset]()

""")
e = st.sidebar.empty()
e.write("")
st.sidebar.write("Brought to you by [Eric Thien, CFA](https://www.linkedin.com/in/eric-thien-cfa-9b96211/)")
#------------------------------------------------------------------------------------------------------------------------------------------------------