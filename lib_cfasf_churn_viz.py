#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Library Imports '''
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import streamlit as st
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Separate columns into demographic info, membership info, employment info and cluster info
col_demographics = ['person_id', 'gender', 'age', 'age_range', 'mailing_locality', 'is_in_france', ]
col_membership1 = ['type_transaction', 'year_joined', 'year_last_membership', 'duration_membership_years', 'duration_program_years']
col_membership2 = ['is_society_member', 'is_curr_year_member', 'is_yet_to_renew_member', 'code_group_membership_change', 'type_cfai_membership']                 
col_employment = [ 'is_on_professional_leave', 'employment_status', 'employer_type', 'seniority']
viz_features_demo = ['gender', 'age', 'locality']
viz_features_all = viz_features_demo + col_membership1 + col_membership2 + col_employment
dict_viz_churn_features = {0: ' ', 1: 'Year Joined', 2: 'Gender', 3: 'Age Distrib.', 4: 'Locality', 
                      5: 'Year Last Membership', 6: 'CFAI Membership Type', 7: 'Duration CFA Program', 8: 'Duration Society Membership', 9: 'Society Membership Change', 
                      10: 'Professional Leave', 11: 'Employment Status', 12:'Employer Types', 13: 'Seniority'}

red_gradient = [
    '#8B0000',  # DarkRed
    '#B22222',  # Firebrick
    '#CD5C5C',  # Indian Red
    '#DC143C',  # Crimson
    '#FF0000',  # Red
    '#FF6347',  # Tomato
    '#FF7F7F',  # Light Coral
    '#FA8072',  # Salmon
    '#FFA07A',  # Light Salmon
    '#FFC0CB'   # Pink (lightest red tone)
]
#------------------------------------------------------------------------------------------------------------------------------------------------------
def get_eda_features():
    return viz_features_all

def get_churn_features():
    viz_churn_features = list(dict_viz_churn_features.values())
    # print(viz_churn_features)
    return viz_churn_features
#------------------------------------------------------------------------------------------------------------------------------------------------------
def fetch_eda_breakdown(opt, feature, df, obj):
    row_height = 20
    mapped_feature = {
        'age': ['age', 'age_range'],
        'locality': ['is_in_france', 'mailing_locality'],
    }
    def map_value(value):
        result = mapped_feature.get(value, [value])
        return result

    if opt != 'none': obj.write(f"Breakdown by {feature} ({opt.capitalize()})")
    dict_filtered_data = {'all': df, 'member':  df[df['is_society_member']], 'non-member': df[~df['is_society_member']], 'none': df[df['is_society_member'].isin([])] } 
    data_filtered = dict_filtered_data[opt]
    calculated_height = min(100, (len(data_filtered) + 1) * row_height)
    if opt != 'none': 
        for value in map_value(feature):
            breakdown_df =  data_filtered[value].value_counts().sort_index()
            # print(breakdown_df)
            obj.dataframe(breakdown_df, height=calculated_height, width=150)
    return data_filtered
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot distribution with Plotly based on feature type
def display_eda_figure(opt, feature, df, obj):
    dict_filtered_data = {'all': df, 'member':  df[df['is_society_member']], 'non-member': df[~df['is_society_member']], 'none': df[df['is_society_member'].isin([])] } 
    data_filtered = dict_filtered_data[opt]
    figs = []
    #--------------------------------------------------------------------------------------------------------------------------------------------
    if feature == 'gender':
        fig_gender = px.pie(data_filtered, names='gender', title=f'Gender Distribution ({opt.capitalize()})',
                color='gender',  # This maps the gender column to the color scheme
                color_discrete_map={'Female': '#EF553B', 'Male': '#636EFA'}, hole=0.3) 
        fig_gender.update_traces(textinfo='label+value+percent')
        figs.append(fig_gender)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'age':
        fig_age_histo = px.histogram(data_filtered, x='age', title=f'Age Distribution - Histogram ({opt.capitalize()})',
                labels={'age': 'Age'}, color_discrete_sequence=['#636EFA'])  # color_discrete_sequence=['lightblue']  # Custom color        
        fig_age_histo.update_yaxes(title_text='Number of persons')
        fig_age_histo.update_traces(
            texttemplate='%{y}', textposition='inside') # Use the bin count for the text # Position the text outside the bars
        fig_age_histo.update_layout(
            # xaxis={'tickmode':'linear', 'dtick':1},
            bargap=0.01, uniformtext_minsize=6, uniformtext_mode='show',)  # Adjust the gap between bars # Ensure text size is uniform
        figs.append(fig_age_histo)
        #----------------------------------------------------------------------------------------------------------------------------------
        fig_age_boxplot = px.box(data_filtered, y='age', title=f'Age Distribution - Boxplot ({opt.capitalize()})',
                labels={'age': 'Age Distrib.'}, color_discrete_sequence=['#636EFA'])  # color_discrete_sequence=['lightblue']  # Custom color
        figs.append(fig_age_boxplot)
        #----------------------------------------------------------------------------------------------------------------------------------
        # px.bar does not automatically calculate counts from a single categorical column.
        counts_age_range = data_filtered['age_range'].value_counts().sort_index().reset_index()
        counts_age_range.columns = ['age_range', 'count']
        print(counts_age_range)
        fig_age_range = px.bar(counts_age_range, x='age_range', y='count', title=f'Age Range Distribution ({opt.capitalize()})',
            labels={'age_range': 'Age ranges', 'count': 'Number of persons'},  # Axis labels
            color_discrete_sequence=['#636EFA'],  # Custom color
            category_orders={'age_range': ['20-25', '26-35', '36-45', '46-55', '56+']},  # Optional: Order the categories
            # text = 'count' # works only if count exists in the df used or text is added as a column to data_filtered
        )
        print(fig_age_range.data)
        fig_age_range.update_traces(
            text=[y/len(data_filtered)*100 for y in fig_age_range.data[0].y],  # Calculate percentage for each bar
            texttemplate='%{y}<br>(%{text:.1f}%)',  # Use the bin count for the text
            textposition='inside')  # Display counts outside of bars for clarity
        figs.append(fig_age_range)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'locality':
        data_filtered['is_in_france_label'] = data_filtered['is_in_france'].map({False: 'Not in France', True: 'Is in France'})
        fig_isin_france = px.pie(data_filtered, names='is_in_france_label', title=f'Is in France Distribution ({opt.capitalize()})',
                    color='is_in_france_label',  # This maps the Is in France column to the color scheme
                    color_discrete_map={'Not in France': '#EF553B', 'Is in France': '#636EFA'}, hole=0.3) 
        fig_isin_france.update_traces(textinfo='label+value+percent')
        figs.append(fig_isin_france)
        #----------------------------------------------------------------------------------------------------------------------------------
        df_demographics_exfrance = data_filtered[ data_filtered['mailing_locality'] != 'France']
        counts_mailing_locality = df_demographics_exfrance['mailing_locality'].value_counts().reset_index()
        counts_mailing_locality.columns = ['mailing_locality', 'count']
        counts_mailing_locality['percentage'] = round( (counts_mailing_locality['count'] / counts_mailing_locality['count'].sum()) * 100, 1)
        # print(counts_mailing_locality)
        list_locality = df_demographics_exfrance['mailing_locality'].value_counts().index.to_list()
        fig_mailing_loc = px.bar(counts_mailing_locality, x='mailing_locality', y='count', title=f'Mailing Locality Outside France Distribution ({opt.capitalize()})',
            labels={'mailing_locality': 'Mailing Locality', 'count': 'Number of persons'},  # Axis labels
            color = 'mailing_locality', color_discrete_sequence=red_gradient,
            category_orders={'mailing_locality': list_locality},  # Optional: Order the categories
            text = 'count'
            )
        # print(fig_mailing_loc.data)
        for i, trace in enumerate(fig_mailing_loc.data):
            trace.customdata = [[counts_mailing_locality['percentage'][i]]]  # Set customdata with percentage
            trace.texttemplate = '%{y}<br>(%{customdata[0]:.1f}%)'  
        fig_mailing_loc.update_traces(
            textposition='outside',)  # Display counts outside of bars for clarity    
        fig_mailing_loc.update_layout(
            uniformtext_minsize=10, uniformtext_mode='show',
            yaxis={'range':[0, counts_mailing_locality['count'].max() * 1.2]},
            xaxis={'tickangle':30},)
        # fig_mailing_loc.update_yaxes(range=[0, counts_mailing_locality['count'].max() * 1.2])
        figs.append(fig_mailing_loc)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'type_transaction':
        fig_type_transaction = px.pie(data_filtered, names='type_transaction', title=f'Transaction Type ({opt.capitalize()})',
                            color='type_transaction',  # This maps the gender column to the color scheme
                            color_discrete_map={'Renew': '#636EFA', 'Join': '#AB63FA', 'Lapse': '#FFA15A', 'Cancel': '#EF553B', }, hole=0.3)
        fig_type_transaction.update_traces(textinfo='label+value+percent')
        figs.append(fig_type_transaction)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'year_joined':
        counts_year_joined = data_filtered['year_joined'].value_counts().sort_index().reset_index()
        counts_year_joined.columns = ['year_joined', 'count']
        counts_year_joined['percentage'] = (counts_year_joined['count'] / counts_year_joined['count'].sum()) * 100
        n_year_joined = data_filtered.year_joined.max() - data_filtered.year_joined.min() + 1
        fig_year_joined = px.histogram(counts_year_joined, x='year_joined', y='count', nbins=n_year_joined, title=f'Year Joined - Histogram ({opt.capitalize()})',
            labels={'year_joined': 'Year Joined', 'count': 'Number of Persons'},
            color_discrete_sequence=['#636EFA'],  # Custom color
            category_orders={'year_joined': sorted(counts_year_joined['year_joined'].unique())} )
        fig_year_joined.update_traces(
            text=[y/counts_year_joined['count'].sum()*100 for y in fig_year_joined.data[0].y],
            texttemplate='%{y}<br>%{text:.0f}%',  # Use the bin count for the text
            textposition='outside') # , textfont_size = 16,) # Position the text outside the bars
        fig_year_joined.update_layout(
            # height=500, margin=dict(t=100), # Increase height to allow more space for text  # Increase top margin
            bargap=0.1, uniformtext_minsize=8, uniformtext_mode='show', # Adjust the gap between bars # Ensure text size is uniform
            yaxis={'range':[0, max(counts_year_joined['count']) * 1.2]},  # Increase y-axis range to 20% above the highest bar
            xaxis={'tickmode':'linear', 'dtick':1, 'tickangle':45},)  # Set interval between ticks to 10 years # tick0=0,  # Start at 0
        figs.append(fig_year_joined)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'year_last_membership':
        counts_year_last_membership = data_filtered['year_last_membership'].value_counts().sort_index().reset_index()
        counts_year_last_membership.columns = ['year_last_membership', 'count']
        counts_year_last_membership['percentage'] = (counts_year_last_membership['count'] / counts_year_last_membership['count'].sum()) * 100
        n_year_last_membership = data_filtered.year_last_membership.max() - data_filtered.year_last_membership.min() + 1
        fig_year_last_membership = px.histogram(counts_year_last_membership, x='year_last_membership', y='count', nbins=n_year_last_membership, title=f'Year of Last Membership - Histogram ({opt.capitalize()})',
            labels={'year_last_membership': 'Year of Last Membership'},  # Label for y-axis
            color_discrete_sequence=['#636EFA'],  # Custom color
            category_orders={'year_last_membership': sorted(counts_year_last_membership['year_last_membership'].unique())} )
        fig_year_last_membership.update_traces(
            text=[y/counts_year_last_membership['count'].sum()*100 for y in fig_year_last_membership.data[0].y],
            texttemplate='%{y}<br>%{text:.0f}%',  # Use the bin count for the text
            textposition='outside') # , textfont_size = 16,) # Position the text outside the bars
        fig_year_last_membership.update_layout(
            # height=500, margin=dict(t=100), # Increase height to allow more space for text  # Increase top margin
            bargap=0.01, uniformtext_minsize=10, uniformtext_mode='show', # Adjust the gap between bars # Ensure text size is uniform
            yaxis={'range':[0, max(counts_year_last_membership['count']) * 1.2]},  # Increase y-axis range to 20% above the highest bar
            xaxis={'tickmode':'linear', 'dtick':1, 'tickangle':45},)  # Set interval between ticks to 10 years # tick0=0,  # Start at 0
        figs.append(fig_year_last_membership)   
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'duration_membership_years':
        fig_dur_membership = px.box(data_filtered, y='duration_membership_years', title=f'Duration Membership Distrib. - Boxplot ({opt.capitalize()})',
            labels={'duration_membership_years': 'Duration Membership (Years)'},  # Label for y-axis
            # color_discrete_sequence=['#636EFA']  # Custom color
        )
        figs.append(fig_dur_membership)   
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'duration_program_years':
        fig_dur_program = px.box(data_filtered, y='duration_program_years', title=f'Duration Program Distrib. - Boxplot ({opt.capitalize()})',
            labels={'duration_program_years': 'Duration Program (Years)'},  # Label for y-axis
            # color_discrete_sequence=['#636EFA']  # Custom color
        )
        figs.append(fig_dur_program)   
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature in col_membership2 + col_employment[0:1]:
        data_filtered['is_society_member_label'] = data_filtered['is_society_member'].map({False: 'Not member', True: 'Society member'})
        data_filtered['is_curr_year_member_label'] = data_filtered['is_curr_year_member'].map({False: 'Not current', True: 'Is current'})
        data_filtered['is_yet_to_renew_member_label'] = data_filtered['is_yet_to_renew_member'].map({False: 'Renewed or non-member', True: 'Yet-to-renew'})
        data_filtered['code_group_membership_change_label'] = data_filtered['code_group_membership_change'].map({0: 'No change', 1: 'Opted-out', 2: 'Opted-in', 3: 'Migration', })
        data_filtered['is_on_professional_leave_label'] = data_filtered['is_on_professional_leave'].map({False: 'No', True: 'Yes'})
        
        dict_membership2 = {'is_society_member': {'name': 'is_society_member_label', 'title': 'Society Member?', 'color':'is_society_member_label', 'color_discrete_map':{'Not member': '#EF553B', 'Society member': '#636EFA'}}, 
                            'is_curr_year_member': {'name': 'is_curr_year_member_label', 'title': 'Current Year Member?', 'color':'is_curr_year_member_label', 'color_discrete_map':{'Not current': '#EF553B', 'Is current': '#636EFA'}},
                            'is_yet_to_renew_member': {'name': 'is_yet_to_renew_member_label', 'title': 'Yet-to-renew Member?', 'color':'is_yet_to_renew_member_label', 'color_discrete_map':{'Yet-to-renew': '#EF553B', 'Renewed or non-member': '#636EFA'}}, 
                            'code_group_membership_change': {'name': 'code_group_membership_change_label', 'title': 'Membership change?', 'color':'code_group_membership_change_label', 'color_discrete_map':{'No change': '#636EFA', 'Opted-in': '#AB63FA', 'Migration': '#FFA15A', 'Opted-out': '#EF553B', }},
                            'type_cfai_membership':{'name': 'type_cfai_membership', 'title': 'Type of CFA member', 'color':'type_cfai_membership', 'color_discrete_map':{'Regular': '#636EFA', 'Candidate': '#AB63FA', 'Affiliate': '#FFA15A', 'Non-member': '#EF553B', }},
                            'is_on_professional_leave':{'name': 'is_on_professional_leave_label', 'title': 'Is on professsional leave?', 'color':'is_on_professional_leave_label', 'color_discrete_map':{'Yes': '#FFA15A', 'No': '#636EFA'}},
                            # 'employment_status':{'name': 'employment_status', 'title': 'Employment Status', 'color':'employment_status', 'color_discrete_map':{'Employed': '#636EFA', 'Self-employed': '#AB63FA', 'Other': '#FFA15A', 'Student': '#00CC96', 'Retired': '#FECB52','Unemployed': '#EF553B', }},
        }
        fig_pie_membership2 = px.pie(data_filtered, names=dict_membership2[feature]['name'], title=f'{dict_membership2[feature]['title']} ({opt.capitalize()})',
                                    color=dict_membership2[feature]['color'],  # This maps the gender column to the color scheme
                                    color_discrete_map=dict_membership2[feature]['color_discrete_map'], hole=0.3) 
        fig_pie_membership2.update_traces(textinfo='label+value+percent')
        figs.append(fig_pie_membership2)    
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'employment_status':
        fig_employment_status_pie = px.pie(data_filtered, names='employment_status', title=f'Employment Status ({opt.capitalize()})',
                            color='employment_status',  # This maps the gender column to the color scheme
                            color_discrete_map={'Employed': '#636EFA', 'Self-employed': '#AB63FA', 'Other': '#FFA15A', 'Student': '#00CC96', 'Retired': '#FECB52','Unemployed': '#EF553B', }, hole=0.3)
        fig_employment_status_pie.update_traces(textinfo='label+value+percent', textposition='outside',)
        fig_employment_status_pie.update_layout(height=500, margin={'t':200})
        figs.append(fig_employment_status_pie)            
        #----------------------------------------------------------------------------------------------------------------------------------
        counts_employment_status = data_filtered['employment_status'].value_counts().sort_index().reset_index()
        counts_employment_status.columns = ['employment_status', 'count']
        counts_employment_status['percentage'] = (counts_employment_status['count'] / counts_employment_status['count'].sum()) * 100
        fig_employment_status_bar = px.bar(counts_employment_status, y='employment_status', x='count', title=f'Employment Status ({opt.capitalize()})',
            labels={'employment_status': 'Employment Status', 'count': 'Number of persons'},  # Axis labels
            color_discrete_sequence=['#636EFA'],  # Custom color
            category_orders={'employment_status': ['Employed', 'Self-employed', 'Student', 'Retired', 'Unemployed', 'Other']}, )  
        fig_employment_status_bar.update_traces(
            text=[x/counts_employment_status['count'].sum()*100 for x in fig_employment_status_bar.data[0].x],  # Calculate percentage for each bar
            texttemplate='%{x}<br>(%{text:.1f}%)',  # Use the bin count for the text
            textposition='outside',)
        fig_employment_status_bar.update_layout(
            # height=500, margin=dict(t=100), # Increase height to allow more space for text  # Increase top margin
            bargap=0.01, uniformtext_minsize=10, uniformtext_mode='show', # Adjust the gap between bars # Ensure text size is uniform
            yaxis={'tickmode':'linear', 'dtick':1, 'tickangle':0},  # Set interval between ticks to 10 years # tick0=0,  # Start at 0
            xaxis={'range':[0, max(counts_employment_status['count']) * 1.2]},)  # Increase y-axis range to 20% above the highest bar
        figs.append(fig_employment_status_bar)     
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'employer_type':
        df_employment_exother = data_filtered[data_filtered['employer_type'] != 'Other (please specify)']
        df_employment_other = data_filtered[data_filtered['employer_type'] == 'Other (please specify)']
        other_employer_types = df_employment_other['employer_type'].value_counts()
        top_employer_types = df_employment_exother['employer_type'].value_counts().nlargest(10)
        final_employer_types = pd.concat([top_employer_types, other_employer_types]).reset_index()
        final_employer_types.columns = ['employer_type', 'count']
        final_employer_types['percentage'] = (final_employer_types['count'] / final_employer_types['count'].sum()) * 100
        fig_employer_types = px.bar(final_employer_types, x='employer_type', y='count', title='Employer Sector',
            labels={'employer_type': 'Employer Sector', 'count': 'Number of persons'},  # Axis labels
            color_discrete_sequence=['#636EFA'], )  # Custom color # color_discrete_sequence=px.colors.qualitative.Plotly
        fig_employer_types.update_traces(
            text=[y/final_employer_types['count'].sum()*100 for y in fig_employer_types.data[0].y],  # Calculate percentage for each bar
            texttemplate='%{y}<br>%{text:.1f}%',  # Use the bin count for the text
            textposition='outside',)
        fig_employer_types.update_layout(
            height=600, # margin=dict(t=100), # Increase height to allow more space for text  # Increase top margin
            bargap=0.01, uniformtext_minsize=9, uniformtext_mode='show', # Adjust the gap between bars # Ensure text size is uniform
            yaxis={'range':[0, max(final_employer_types['count']) * 1.2]},  # Increase y-axis range to 20% above the highest bar
            xaxis={'dtick':1, 'tickmode':'linear', 'tickangle':30},)  # Set interval between ticks to 10 years # tick0=0,  # Start at 0, 'dtick':1, 
        figs.append(fig_employer_types)      
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif feature == 'seniority':
        counts_seniority = data_filtered['seniority'].value_counts().reset_index()
        counts_seniority.columns = ['seniority', 'count']
        counts_seniority['percentage'] = (counts_seniority['count'] / counts_seniority['count'].sum()) * 100
        fig_job_seniority = px.bar(counts_seniority, x='seniority', y='count', title=f'Job Seniority  ({opt.capitalize()})',
            labels={'seniority': 'Job Seniority', 'count': 'Number of persons'},  # Axis labels
            color_discrete_sequence=['#636EFA'],  # Custom color # color_discrete_sequence=px.colors.qualitative.Plotly
            category_orders={'seniority': ['C-Suite', 'Executive-Level', 'Senior-Level', 'Mid-Level', 'Early/Mid-Level', 'Entry-Level', 'Other']}, ) # Optional: Order the categories
        fig_job_seniority.update_traces(
            text=[y/counts_seniority['count'].sum()*100 for y in fig_job_seniority.data[0].y],  # Calculate percentage for each bar
            texttemplate='%{y}<br>(%{text:.1f}%)',  # Use the bin count for the text
            textposition='outside',)
        fig_job_seniority.update_layout(
            # height=500, margin=dict(t=100), # Increase height to allow more space for text  # Increase top margin
            bargap=0.01, uniformtext_minsize=10, uniformtext_mode='show', # Adjust the gap between bars # Ensure text size is uniform
            yaxis={'range':[0, max(counts_seniority['count']) * 1.2]},  # Increase y-axis range to 20% above the highest bar
            xaxis={'tickmode':'linear', 'dtick':1, 'tickangle':45},)  # Set interval between ticks to 10 years # tick0=0,  # Start at 0
        figs.append(fig_job_seniority)         
    #--------------------------------------------------------------------------------------------------------------------------------------------    
    else:
        fig = px.histogram(data_filtered, x=feature, color='churned', barmode='overlay',
                           title=f"{feature.capitalize()} Breakdown by Churn Status")
        fig.update_layout(bargap=0.2, template='plotly_white', plot_bgcolor='white', paper_bgcolor='white')
    #--------------------------------------------------------------------------------------------------------------------------------------------
    for fig in figs:
        obj.plotly_chart(fig, use_container_width=True)
    return figs

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot distribution with Plotly based on churn feature type
def display_churn_figure(i_feature, filtered_data, obj):
    figs = []; commentary1 = ''; commentary2 = ''; commentary3 = ''
    #--------------------------------------------------------------------------------------------------------------------------------------------
    if dict_viz_churn_features[i_feature] == 'Year Joined':
        df = filtered_data.groupby(['year_joined', 'churned']).size().reset_index(name='counts')
        df['total_counts'] = df.groupby(['year_joined'])['counts'].transform('sum')
        df['percentage'] = round( (df['counts'] / df['total_counts']) * 100, 2)
        df['avr_pct'] =  round(df.groupby(['churned'])['percentage'].transform('mean'), 2)
        n_year_joined = filtered_data.year_joined.max() - filtered_data.year_joined.min() + 1
        non_churned = df[df['churned'] == 0]
        churned = df[df['churned'] == 1]    
        fig1 = go.Figure(
            data=[
                go.Bar(
                    x=non_churned['year_joined'].astype(str), y=non_churned['counts'], name='Non-Churned', 
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(non_churned['counts'], non_churned['percentage'])], textposition='outside', textangle = 0, 
                    marker_color='lightgreen' ),
                go.Bar(
                    x=churned['year_joined'].astype(str), y=churned['counts'], name='Churned', 
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(churned['counts'], churned['percentage'])], textposition='outside', textangle = 0, 
                    marker_color='salmon' )
                ],      )
        # fig1.update_yaxes(range=[-30,250])
        fig1.update_layout(
            barmode='stack', uniformtext_minsize=12, uniformtext_mode='show', # barmode='overlay'
            template='plotly_white', plot_bgcolor='white', paper_bgcolor='white',
            title='Churned vs Non-Churned by Year Joined', legend_title='Churn Status', # title="Year Joined Impact on Churn
            yaxis={'title':'Member Count','range':[-30, df['total_counts'].max() * 1.2]},
            xaxis={'title':'Year Joined (Vintage)', 'tickangle':0}, )
        figs.append(fig1)       
        commentary1 = '''<br>Total Churn (Non-renewal) rate  usually varies between 50-70% per vintage.
                        <br>Churn rates are usually lower for vintages < 5 years ~30-60%'''
        #----------------------------------------------------------------------------------------------------------------------------------
        renew_2024 = (filtered_data['type_transaction']=='Renew') & (filtered_data['year_last_membership']==2024) 
        renew_2025 = (filtered_data['type_transaction']=='Renew') & (filtered_data['year_last_membership']==2025) 
        filtered_data.loc[renew_2024, 'type_transaction'] = 'Renew 2024'
        filtered_data.loc[renew_2025, 'type_transaction'] = 'Renew 2025'
        df2 = filtered_data.groupby(['year_joined', 'type_transaction']).size().reset_index(name='counts')
        df2['total_counts'] = df2.groupby(['year_joined'])['counts'].transform('sum')
        df2['percentage'] = round( (df2['counts'] / df2['total_counts']) * 100, 2)
        df2['avr_pct'] =  round(df2.groupby(['type_transaction'])['percentage'].transform('mean'), 2)
        cancelled = df2[df2['type_transaction'] == 'Cancel']
        lapsed = df2[df2['type_transaction'] == 'Lapse']
        joined = df2[df2['type_transaction'] == 'Join']
        renewed_last_year = df2[df2['type_transaction'] == 'Renew 2024']
        renewed_this_year = df2[df2['type_transaction'] == 'Renew 2025']
        fig2 = go.Figure(
            data=[
                go.Bar(
                    x=renewed_this_year['year_joined'].astype(str), y=renewed_this_year['counts'], name='Renewed 2025', 
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(renewed_this_year['counts'], renewed_this_year['percentage'])], textposition='outside', textangle = 0, 
                    marker_color='lightgreen' ),
                go.Bar(
                    x=renewed_last_year['year_joined'].astype(str), y=renewed_last_year['counts'], name='Renewed 2024', 
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(renewed_last_year['counts'], renewed_last_year['percentage'])], textposition='outside', textangle = 0, 
                    marker_color='#636EFA' ),
                go.Bar(
                    x=joined['year_joined'].astype(str), y=joined['counts'], name='Joined', 
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(joined['counts'], joined['percentage'])], textposition='outside', textangle = 0, 
                    marker_color='#AB63FA' ),
                go.Bar(
                    x=lapsed['year_joined'].astype(str), y=lapsed['counts'], name='Lapsed', 
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(lapsed['counts'], lapsed['percentage'])], textposition='outside', textangle = 0, 
                    marker_color='#FFA15A' ),
                go.Bar(
                    x=cancelled['year_joined'].astype(str), y=cancelled['counts'], name='Cancelled', 
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(cancelled['counts'], cancelled['percentage'])], textposition='outside', textangle = 0, 
                    marker_color='#EF553B' ),
                ], )
        fig2.update_layout(
            barmode='stack', uniformtext_minsize=10, uniformtext_mode='show', # barmode='overlay'
            template='plotly_white', plot_bgcolor='white', paper_bgcolor='white',
            title='Transaction Type by Year Joined', legend_title='Churn Status', # title="Year Joined Impact on Churn
            yaxis={'title':'Member Count','range':[-30, df['total_counts'].max() * 1.2]},
            xaxis={'title':'Year Joined (Vintage)', 'tickangle':0}, )
        figs.append(fig2)     
        #----------------------------------------------------------------------------------------------------------------------------------
        df['churned_label'] = df['churned'].map({False: 'Non-Churned', True: 'Churned'})
        fig3 = px.histogram(df, x='year_joined', y='counts', barmode='stack', nbins=n_year_joined, 
                        title='Year Joined vs Churned', labels={'churned': 'Churn Status'},
                        color='churned_label', color_discrete_map={'Non-Churned': '#EF553B', 'Churned': '#636EFA'},
                        text_auto=True, )
        # fig3.for_each_trace(lambda t: t.update(name='Churned' if t.name == "True" else 'Non-Churned'))
        # Add average percentage traces (line graphs)
        for churn_status, color, color2 in [(0, 'cornflowerblue', 'blue'), (1, 'tomato', 'red')]:
            avg_df = df[df['churned'] == churn_status]
            fig3.add_trace(go.Scatter(x=avg_df['year_joined'], y=avg_df['percentage'], name=f'Pct {"Non-Churned" if churn_status == 0 else "Churned"}',
                        mode='lines+markers', line=dict(color=color, width=3), yaxis='y2')) #, hovertemplate='%{y}%')) dash='dash', 
            fig3.add_trace(go.Scatter(x=avg_df['year_joined'], y=avg_df['avr_pct'], name=f'Avr Pct {"Non-Churned" if churn_status == 0 else "Churned"}',
                        mode='lines+markers', line=dict(color=color2, dash='dash', width=3), yaxis='y2')) #, hovertemplate='%{y}%')) dash='dash', 
        fig3.update_layout(
            barmode='stack', bargap=0.0, uniformtext_minsize=12, uniformtext_mode='show',
            template='plotly_white', plot_bgcolor='white', paper_bgcolor='white',
            title='Churned vs Non-Churned by Year Joined (Comparison with Averages)', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts'].max() * 1.2]},
            yaxis2={'title':'Percentage', 'overlaying':'y', 'side':'right', 'range':[0, 100], 'showgrid':False},
            xaxis={'title':'Year Joined (Vintage)', 'tickangle':0, 'tickmode': 'linear'}, )
        figs.append(fig3)    
        commentary3 = '''<br>Average churn rate (across all years) : ~61.6%
                        <br>Average non-churn rate (across all years) : ~38.4%
                        <br>Non-churn rate lowest for last 5 years (recent members)'''
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Gender':
        color_map = { ('Male', 0): 'lightblue', ('Male', 1): 'skyblue', ('Female', 0): 'pink', ('Female', 1): 'salmon' }    
        #------------------------------------------------------------------------------------------------------------------------
        df = filtered_data.groupby(['churned', 'gender']).size().reset_index(name='counts')
        df['total_counts_per_gender'] = df.groupby('gender')['counts'].transform('sum')
        df['percentage'] = (df['counts'] / df['total_counts_per_gender']) * 100
        fig1 = go.Figure()
        for i, row in df.iterrows():
            fig1.add_trace(go.Bar(
                x=[row['gender']], y=[row['counts']], name=f"{row['gender']} - {'Churned' if row['churned'] else 'Non-Churned'}",
                text=f"{row['counts']} ({row['percentage']:.1f}%)", textposition='inside',
                marker_color=color_map[(row['gender'], row['churned'])] 
            ))
        fig1.update_layout(
            barmode='stack', uniformtext_minsize=10, uniformtext_mode='show', # barmode='overlay'
            template='plotly_white', plot_bgcolor='white', paper_bgcolor='white',
            title='Churned vs Non-Churned by Gender', legend_title='Churn Status', # title="Year Joined Impact on Churn
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_gender'].max() * 1.2]},
            xaxis={'title':'Gender', 'tickangle':0}, )
        figs.append(fig1)   
        commentary1 = '''<br>No significant difference in churn rates between genders (across the board).'''
        #------------------------------------------------------------------------------------------------------------------------
        df2 = filtered_data.groupby(['year_joined', 'churned', 'gender']).size().reset_index(name='counts')
        df2['total_counts_per_gender_year'] = df2.groupby(['year_joined', 'gender'])['counts'].transform('sum')
        df2['percentage'] = (df2['counts'] / df2['total_counts_per_gender_year']) * 100
        fig2 = go.Figure()
        for gender in ['Male', 'Female']:
            for churn in [0, 1]:
                df_filtered = df2[(df2['gender'] == gender) & (df2['churned'] == churn)]
                fig2.add_trace(go.Bar(
                    x=df_filtered['year_joined'].astype(str) + ' (' + df_filtered['gender'] + ')', y=df_filtered['counts'],
                    name=f"{gender} - {'Churned' if churn else 'Non-Churned'}",
                    text=[f"{c}<br>({p:.1f}%)" for c, p in zip(df_filtered['counts'], df_filtered['percentage'])], 
                    textposition='outside', textangle = 315, textfont_size=8, cliponaxis=False,
                    marker_color=color_map.get((gender, churn)), ))        
        fig2.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=10, uniformtext_mode='show', # barmode='overlay'
            template='plotly_white', plot_bgcolor='white', paper_bgcolor='white',
            title='Churned vs Non-Churned by Gender and Year Joined', legend_title='Gender Churn Status', # title="Year Joined Impact on Churn
            yaxis={'title':'Member Count','range':[-30, df2['total_counts_per_gender_year'].max() * 1.2]},
            xaxis={'title':'Year Joined (Gender)', 'categoryorder':'category ascending', 'tickangle':30}, )
        figs.append(fig2)   
        commentary2 = '''<br>Some notable differences when zoomed-in by Year Joined.'''
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Age Distrib.':
        df = filtered_data
        df['churned_label'] = df['churned'].map({False: 'Non-Churned', True: 'Churned'})
        fig1 = px.box(df, x='churned_label', y='age', color='churned_label',
                    title='Age Distribution by Churn Status', labels={"churned_label": "Churn Status", "age": "Age"},
                    color_discrete_map={'Churned': '#EF553B', 'Non-Churned': '#636EFA'},
                    category_orders={"churned": ['Non-Churned', 'Churned']}, 
                    )
        # fig1.for_each_trace(lambda t: t.update(name=t.name.replace("True", "Churned").replace("False", "Non-Churned")))
        fig1.update_traces(marker={'line':{'width':2}})
        figs.append(fig1)  
        commentary1='''<br>No significant difference in age distribution between churn & non-churn.'''
        #------------------------------------------------------------------------------------------------------------------------
        df2 = filtered_data.groupby(['churned', 'age_range']).size().reset_index(name='counts')
        df2['total_counts_per_agegroup'] = df2.groupby('age_range')['counts'].transform('sum')
        df2['percentage'] = round( (df2['counts'] / df2['total_counts_per_agegroup']) * 100, 2)
        fig2 = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: # zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df2[df2['churned'] == churn_status]
            fig2.add_trace(go.Bar(
                x=churn_data['age_range'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c} ({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside',
                marker_color=color, )) # ('#EF553B' if churn_status else '#636EFA'), ))   
        fig2.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=14, uniformtext_mode='show', # barmode='overlay'
            # template='plotly_white', plot_bgcolor='white', paper_bgcolor='white',
            title='Churned vs Non-Churned by Age Group', legend_title='Churn Status', # title="Year Joined Impact on Churn
            yaxis={'title':'Member Count','range':[-30, df2['total_counts_per_agegroup'].max() * 1.2]},
            xaxis={'title':'Age Group', 'categoryorder':'category ascending', 'tickangle':0}, )
        figs.append(fig2)
        commentary2='''<br>Biggest chrun in 36-45 age group where most members are in.
                        <br>Median age of society member is 40.'''  
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Locality':
        df = filtered_data.groupby(['churned', 'mailing_locality']).size().reset_index(name='counts')
        df['total_counts_per_locality'] = df.groupby('mailing_locality')['counts'].transform('sum')
        df['percentage'] = round( (df['counts'] / df['total_counts_per_locality']) * 100, 2)
        custom_order = ['France', 'Europe excl France', 'United Kingdom', 'Asia-Pacific', 'North Africa', 'Sub-Saharan Africa', 'Middle East', 'North America', 'Latin America']  # Add any other localities as needed
        df['mailing_locality'] = pd.Categorical(df['mailing_locality'], categories=custom_order, ordered=True)
        df = df.sort_values('mailing_locality')
        fig = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: # zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df[df['churned'] == churn_status]
            fig.add_trace(go.Bar(
                x=churn_data['mailing_locality'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c} ({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside',
                marker_color=color, )) # ('#EF553B' if churn_status else '#636EFA'), ))   
        fig.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=14, uniformtext_mode='show', # barmode='overlay'
            title='Churned vs Non-Churned by Locality', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_locality'].max() * 1.2]},
            xaxis={'title':'Locality', 'categoryorder':'array', 'tickangle':0}, )  # 'categoryorder':'category ascending'
        figs.append(fig)
        commentary1='''<br>Higher churn when not in France, as can be expected.''' 
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Year Last Membership':
        df = filtered_data.groupby(['year_last_membership', 'churned']).size().reset_index(name='counts')
        df['total_counts_per_lastyearmembership'] = df.groupby('year_last_membership')['counts'].transform('sum')
        year_range = pd.DataFrame({'year_last_membership': list(range(2000, 2026))})
        df = pd.merge(year_range, df, on='year_last_membership', how='left').fillna(0)
        df.loc[df['churned'] == 0, 'churned'] = False
        df['total_count'] = df['counts'].sum()
        df['percentage'] = round((df['counts'] / df['total_count']) * 100, 2)
        fig = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: # zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df[df['churned'] == churn_status]
            fig.add_trace(go.Bar(
                x=churn_data['year_last_membership'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c}<br>({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside', #textfont_size=11, textangle = 0, 
                marker_color=color, )) # ('#EF553B' if churn_status else '#636EFA'),
        fig.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=14, uniformtext_mode='show', # barmode='overlay'
            title='Churned vs Non-Churned by Year of Last Membership', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_lastyearmembership'].max() * 1.2]},
            xaxis={'title':'Year Last Membership', 'categoryorder':'category ascending', 'tickmode': 'linear', 'tickangle':0}, )  
        figs.append(fig)
        commentary1='''<br>Churn seems to be on an increasing trend... 
                        <br>Cf. Year 2023 with 215 members churned, representing 9.2% of the total population.''' 
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'CFAI Membership Type':
        df = filtered_data.groupby(['churned', 'type_cfai_membership']).size().reset_index(name='counts')
        df['total_counts_per_cfaimembershiptype'] = df.groupby('type_cfai_membership')['counts'].transform('sum')
        df['percentage'] = round( (df['counts'] / df['total_counts_per_cfaimembershiptype']) * 100, 2)
        custom_order = ['Regular', 'Affiliate', 'Candidate', 'Non-member']
        df['type_cfai_membership'] = pd.Categorical(df['type_cfai_membership'], categories=custom_order, ordered=True)
        df = df.sort_values('type_cfai_membership')
        fig = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: # zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df[df['churned'] == churn_status]
            fig.add_trace(go.Bar(
                x=churn_data['type_cfai_membership'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c} ({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside', #textfont_size=11, textangle = 0, 
                marker_color= color, )) # ('#EF553B' if churn_status else '#636EFA'),
        fig.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=14, uniformtext_mode='show', # barmode='overlay'
            title='Churned vs Non-Churned by CFAI Membership Type', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_cfaimembershiptype'].max() * 1.2]},
            xaxis={'title':'CFAI Membership Type', 'categoryorder':'array', 'tickmode': 'linear', 'tickangle':0}, )  
        figs.append(fig)
        commentary1='''<br>Less churn when you’re a CFAI regular member!''' 
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Duration CFA Program':
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Separate Violin Plots for Churned vs Non-Churned", "Superposed Violin Plot"))
        # Violin plot for churned = 0 (non-churned) and churned = 1 (churned) in separate graphs
        fig.add_trace(go.Violin(x=filtered_data[ filtered_data['churned'] == 1]['churned'].map({True: 'Churned', False: 'Non-Churned'}),
                                y=filtered_data['duration_program_years'][filtered_data['churned'] == 1],
                                name='Churned', line_color='red'), row=1, col=1)
        fig.add_trace(go.Violin(x=filtered_data[ filtered_data['churned'] == 0]['churned'].map({True: 'Churned', False: 'Non-Churned'}),
                                y=filtered_data['duration_program_years'][filtered_data['churned'] == 0],
                                name='Non-Churned', line_color='blue'), row=1, col=1)
        fig.add_trace(
            px.violin(filtered_data[filtered_data['churned'] == 1], y="duration_program_years", 
                    color_discrete_sequence=['red']).data[0], row=1, col=2)
        fig.add_trace(
            px.violin(filtered_data[filtered_data['churned'] == 0], y="duration_program_years", 
                    color_discrete_sequence=['blue']).data[0], row=1, col=2)
        fig.update_layout(
            violinmode='overlay', # violingap=0.5,  # gap between violins
            height=600, width=1200,
            title_text="Violin Plots of CFA Program Duration (Years) vs Churned Status", legend_title='Churn Status', showlegend=True, 
            yaxis={'title':'Program Duration (Years)'}, 
            xaxis={'title':'Churned vs Non-Churned', 'tickmode':'linear', 'tickangle':0}, 
            xaxis2={'title':'Superposed'}, )
        figs.append(fig)
        commentary1='''<br>More investment in the program seems to suggest lower churn...'''
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Duration Society Membership':
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Separate Violin Plots for Churned vs Non-Churned", "Superposed Violin Plot"))
        fig.add_trace(go.Violin(x=filtered_data[ filtered_data['churned'] == 1]['churned'].map({True: 'Churned', False: 'Non-Churned'}),
                                y=filtered_data['duration_membership_years'][filtered_data['churned'] == 1],
                                name='Churned', line_color='red'), row=1, col=1)
        fig.add_trace(go.Violin(x=filtered_data[ filtered_data['churned'] == 0]['churned'].map({True: 'Churned', False: 'Non-Churned'}),
                                y=filtered_data['duration_membership_years'][filtered_data['churned'] == 0],
                                name='Non-Churned', line_color='blue'), row=1, col=1)
        fig.add_trace(
            px.violin(filtered_data[filtered_data['churned'] == 1], y="duration_membership_years", 
                    color_discrete_sequence=['red']).data[0], row=1, col=2)
        fig.add_trace(
            px.violin(filtered_data[filtered_data['churned'] == 0], y="duration_membership_years", 
                    color_discrete_sequence=['blue']).data[0], row=1, col=2)
        fig.update_layout(
            violinmode='overlay', # violingap=0.5,  # gap between violins
            height=600, width=1200,
            title_text="Violin Plots of Society Membership Duration (Years) vs Churned Status", legend_title='Churn Status', showlegend=True, 
            yaxis={'title':'Society Membership Duration (Years)'}, 
            xaxis={'title':'Churned vs Non-Churned', 'tickmode':'linear', 'tickangle':0}, 
            xaxis2={'title':'Superposed'}, )
        figs.append(fig)
        commentary1='''<br>After ~7 years, the longer you stay as a society member, the less likely you are to churn!'''
    #--------------------------------------------------------------------------------------------------------------------------------------------    
    elif dict_viz_churn_features[i_feature] == 'Society Membership Change':
        df = filtered_data
        df.loc[df['type_transaction'] == 'Lapse', 'reason_group_membership_change'] = 'Expired'
        df = filtered_data.groupby(['churned', 'reason_group_membership_change']).size().reset_index(name='counts')
        df['total_counts_per_reasoncategory'] = df.groupby('reason_group_membership_change')['counts'].transform('sum')
        df['percentage'] = round( (df['counts'] / df['total_counts_per_reasoncategory']) * 100, 2)
        custom_order = ['Expired', 'Opt-out', 'Move in to this society', 'Move out to other society']
        df['reason_group_membership_change'] = pd.Categorical(df['reason_group_membership_change'], categories=custom_order, ordered=True)
        df = df.sort_values('reason_group_membership_change')
        fig1 = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: # zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df[df['churned'] == churn_status]
            fig1.add_trace(go.Bar(
                x=churn_data['reason_group_membership_change'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c} ({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside', # textfont_size=11, textangle = 0, 
                marker_color=color ))
        fig1.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=14, uniformtext_mode='show', # barmode='overlay'
            title='Churned vs Non-Churned by Reason Category', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_reasoncategory'].max() * 1.2]},
            xaxis={'title':'Membership Change Group Category', 'categoryorder':'array', 'tickmode': 'array', 'tickangle':0}, )  
        figs.append(fig1)  
        commentary1='''<br>Many members let their memberships lapse.
                        <br>Quite a lot cancelled all the same.'''
        #------------------------------------------------------------------------------------------------------------------------
        df2 = filtered_data
        df2.loc[df2['type_transaction'] == 'Lapse', 'reason_group_membership_change'] = 'Expired'
        df2 = filtered_data.groupby(['year_last_membership', 'reason_group_membership_change']).size().reset_index(name='counts')
        year_range = pd.DataFrame({'year_last_membership': list(range(2000, 2026))})
        df2 = pd.merge(year_range, df2, on='year_last_membership', how='left').fillna(0)
        df2.loc[df2['reason_group_membership_change'] == 0, 'reason_group_membership_change'] = 'No reason'
        df2['total_counts_per_reasoncategory'] = df2.groupby('year_last_membership')['counts'].transform('sum')
        df2['percentage'] = round((df2['counts'] / df2['total_counts_per_reasoncategory']) * 100, 2)
        custom_order = ['Expired', 'Opt-out', 'Move in to this society', 'Move out to other society']
        df2['reason_group_membership_change'] = pd.Categorical(df2['reason_group_membership_change'], categories=custom_order, ordered=True)
        df2 = df2.sort_values('reason_group_membership_change')
        fig2 = go.Figure()
        for churn_status, color in [('Expired', 'brown'), ('Opt-out', '#EF553B'), ('Move in to this society', '#00CC96'), ('Move out to other society','#AB63FA')]:
        # for churn_status, color in zip(df2['reason_group_membership_change'].unique(), ('brown', '#EF553B', '#00CC96', '#AB63FA')):
            churn_data = df2[df2['reason_group_membership_change'] == churn_status]
            fig2.add_trace(go.Bar(
                x=churn_data['year_last_membership'], y=churn_data['counts'], name=f"{churn_status}",
                text=[f"{c}<br>({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside', textfont_size=11, textangle = 0, 
                marker_color=color ))
        fig2.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=12, uniformtext_mode='show', # barmode='overlay'
            title='Society Membership Change by Reason Category and Breakdown by Year Joined', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df2['total_counts_per_reasoncategory'].max() * 1.2]},
            xaxis={'title':'Year Joined', 'categoryorder':'category ascending', 'tickmode': 'linear', 'tickangle':0}, )  
        figs.append(fig2)
        commentary2='''<br>Cf. Year 2023, of the 215 members churned : 
                        <br>*24 left for other societies (11.2%),* 
                        <br>*87 cancelled for various reasons (40.5%),* 
                        <br>*104 lapsed (48.4%)*

                        Reason Categories: [1: Opted-out (Cancelled) | 2: Opted-in | 3: Migrated]
                        ''' 
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Professional Leave':
        df = filtered_data.groupby(['is_on_professional_leave', 'churned']).size().reset_index(name='counts')
        churn_labels = {0: 'Non-Churned', 1: 'Churned'}; colors = {0: '#636EFA', 1: '#EF553B'}

        fig = go.Figure()
        for i, leave_status in enumerate([True, False]):
            df_filtered = df[df['is_on_professional_leave'] == leave_status]   
            fig.add_trace(go.Pie(
                values=df_filtered['counts'], labels=[churn_labels[churn] for churn in df_filtered['churned']], 
                name=f'Professional Leave = {leave_status}', title=f'Leave Status: {leave_status}',
                textinfo=f'label+value+percent', marker=dict(colors=[colors[label] for label in churn_labels]), textfont_size=12,
                domain=dict(x=[i * 0.5, (i + 1) * 0.5]), hole=0.3, )) # Donut style
        fig.update_layout(
            title_text="Churned vs Non-Churned for Professional Leave Status", legend_title='Churn Status', showlegend=True, 
            annotations=[dict(text='Is on Profession Leave', x=0.20, y=-0.15, font_size=15, showarrow=False),
                        dict(text='Not on Profession Leave', x=0.81, y=-0.15, font_size=15, showarrow=False)],
            grid=dict(rows=1, columns=2), )
        figs.append(fig)
        commentary1='<br>Higher churn when you’re on professional leave, as expected.'
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Employment Status':
        df = filtered_data.groupby(['churned', 'employment_status']).size().reset_index(name='counts')
        df['total_counts_per_employmentstatus'] = df.groupby('employment_status')['counts'].transform('sum')
        df['percentage'] = round( (df['counts'] / df['total_counts_per_employmentstatus']) * 100, 2)
        custom_order = ['Employed', 'Self-employed', 'Student', 'Retired', 'Unemployed', 'Other']
        df['employment_status'] = pd.Categorical(df['employment_status'], categories=custom_order, ordered=True)
        df = df.sort_values('employment_status')
        fig = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: # zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df[df['churned'] == churn_status]
            fig.add_trace(go.Bar(
                x=churn_data['employment_status'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c} ({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside', textfont_size=11, textangle=0, 
                marker_color=color, ))
        fig.update_layout(
            barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=12, uniformtext_mode='show', # barmode='overlay'
            title='Churned vs Non-Churned by Employment Status', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_employmentstatus'].max() * 1.2]},
            xaxis={'title':'Employment Status Type', 'categoryorder':'array', 'tickmode': 'linear', 'tickangle':0}, )  
        figs.append(fig)
        commentary1='''<br>Higher churn when not employed or self-employed.'''      
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Employer Types':
        df = filtered_data.groupby(['churned', 'employer_type']).size().reset_index(name='counts')
        df['total_counts_per_employertype'] = df.groupby('employer_type')['counts'].transform('sum')
        df['percentage'] = round( (df['counts'] / df['total_counts_per_employertype']) * 100, 2)
        df_sorted = df.groupby('employer_type').agg({'counts': 'sum'}).reset_index().sort_values('counts', ascending=False)
        employer_order = df_sorted['employer_type'].tolist(); str_other = 'Other (please specify)'
        employer_order.remove(str_other); employer_order.append(str_other)
        df['employer_type'] = pd.Categorical(df['employer_type'], categories=employer_order, ordered=True)
        df = df.sort_values(['employer_type', 'churned'])
        fig = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: # zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df[df['churned'] == churn_status]
            fig.add_trace(go.Bar(
                x=churn_data['employer_type'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c}<br>({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside', textfont_size=11, textangle = 0, 
                marker_color=color ))
        fig.update_layout(
            height=600, barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=12, uniformtext_mode='show', # barmode='overlay'
            title='Churned vs Non-Churned by Employment Type', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_employertype'].max() * 1.2]},
            xaxis={'title':'Employer Type', 'categoryorder':'array', 'tickmode': 'linear', 'tickangle':15}, )  
        figs.append(fig)
        commentary1='''<br>Less churn when you’re in asset management! Bankers and consultants are significant in numbers too.'''      
    #--------------------------------------------------------------------------------------------------------------------------------------------
    elif dict_viz_churn_features[i_feature] == 'Seniority':
        df = filtered_data.groupby(['churned', 'seniority']).size().reset_index(name='counts')
        df['total_counts_per_seniority'] = df.groupby('seniority')['counts'].transform('sum')
        df['percentage'] = round( (df['counts'] / df['total_counts_per_seniority']) * 100, 2)
        custom_order = ['C-Suite', 'Executive-Level', 'Senior-Level', 'Mid-Level', 'Early/Mid-Level','Entry-Level', 'Other']
        df['seniority'] = pd.Categorical(df['seniority'], categories=custom_order, ordered=True)
        df = df.sort_values('seniority')
        fig = go.Figure()
        for churn_status, color in [(0, '#636EFA'), (1, '#EF553B')]: #zip(df['churned'].unique(), ('#EF553B', '#636EFA')):
            churn_data = df[df['churned'] == churn_status]
            fig.add_trace(go.Bar(
                x=churn_data['seniority'], y=churn_data['counts'], name=f"{'Churned' if churn_status else 'Non-Churned'}",
                text=[f"{c}<br>({p:.1f}%)" for c, p in zip(churn_data['counts'], churn_data['percentage'])], textposition='outside', textfont_size=11, textangle = 0, 
                marker_color=color ))
        fig.update_layout(
            height=600, barmode='stack', bargap=0.0, bargroupgap=0.3, uniformtext_minsize=12, uniformtext_mode='show', # barmode='overlay'
            title='Churned vs Non-Churned by Seniority', legend_title='Churn Status',
            yaxis={'title':'Member Count','range':[-30, df['total_counts_per_seniority'].max() * 1.2]},
            xaxis={'title':'Seniority', 'categoryorder':'array', 'tickmode': 'linear', 'tickangle':0}, )  
        figs.append(fig)
        commentary1='''<br>C-Suite members churn less.'''    
    #--------------------------------------------------------------------------------------------------------------------------------------------
    else:
        obj.markdown("***:red[Click on any tab to examine feature impact on churn.]***") 
    #--------------------------------------------------------------------------------------------------------------------------------------------
    for i, fig in enumerate(figs):
        obj.plotly_chart(fig, use_container_width=True)
        commentary = eval(f'commentary{str(eval('i+1'))}')
        if commentary: obj.write(f"**Commentary** : {commentary}", unsafe_allow_html=True)
