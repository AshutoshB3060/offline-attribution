# Databricks notebook source
# MAGIC %md 
# MAGIC ## Objective of this Notebook:
# MAGIC * Prepare training and test datasets.
# MAGIC   * Import datasets from excel, csv files into pandas dataframes by using the excel, csv connectors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC ### 1. <a href='#InstallDependencies'>Install Dependencies</a>
# MAGIC ### 2. <a href='#ImportMultipleDataframes'>Import Multiple Dataframes</a>
# MAGIC ### 3. <a href='#MakeaSingleDataframe'>Make a Single Dataframe</a>

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Install Dependencies <a id='InstallDependencies'></a>

# COMMAND ----------

# # Install all the required libraries

# dbutils.library.installPyPI('mlflow', version='2.2.2')
# dbutils.library.installPyPI('openpyxl', version='3.1.2')
# dbutils.library.restartPython()

# COMMAND ----------

# # Configure widgets

# dbutils.widgets.removeAll()
# dbutils.widgets.text("file_location", "/dbfs/FileStore/FileStore/offline_attribution/datasets/")
# file_location = dbutils.widgets.getArgument(name='file_location')
# dbutils.widgets.text("experiment_path", "/Shared/Analytics/offline_attribution/offline_attribution_experiments")
# experiment_path = dbutils.widgets.getArgument(name='experiment_path')

# COMMAND ----------

# Import all the necessary libraries

import numpy as np
import pandas as pd
import mlflow
from  mlflow.tracking import MlflowClient
import pprint
pp = pprint.PrettyPrinter(indent=4)
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# # # Create new experiment (Only for the first time)
# # experiment_id = mlflow.create_experiment("/Shared/Analytics/offline_attribution/offline_attribution_experiments")
# # experiment = mlflow.get_experiment(experiment_id)

# client = MlflowClient()
# experiment = client.get_experiment_by_name(experiment_path)
# ex_id = experiment.experiment_id

# COMMAND ----------

# Display settings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Import Multiple Dataframes<a id="ImportMultipleDataframes"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC * Broadly, there are 4 types of datasets available
# MAGIC 1. TV
# MAGIC 2. OOH
# MAGIC 3. External features
# MAGIC 4. Attributed Channels

# COMMAND ----------

# Declare the file name for different datasets

tv_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/tv_dataset_offline_attribution.csv'
ooh_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/ooh_dataset_offline_attribution.csv'
ext_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/external_variables_offline_attribution.xlsx'
snf_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/snowflake_features_offline_attribution.csv'
crm_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/crm_dataset_offline_attribution.csv'
prtnrshps_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/partnerships_dataset_offline_attribution.csv'
prtnrshps_incntv_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/partnership_incentive_data.csv'
afflts_file = '/dbfs/FileStore/FileStore/offline_attribution/datasets/affiliates_data.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC * TV dataset

# COMMAND ----------

df_tv = pd.read_csv(tv_file)
dict_tv_rename_cols = {'Date':'tv_date', 'Client Cost(GBP)':'tv_cost', 'Day of Week':'day_of_week', 'Equivalent Impacts (All Adults)':'tv_impact_adult'}
df_tv.rename(columns=dict_tv_rename_cols, inplace=True)
df_tv.columns = df_tv.columns.str.lower()
df_tv['tv_date'] = pd.to_datetime(df_tv['tv_date'])
df_tv_grouped = df_tv.groupby('tv_date', as_index=False)['tv_cost'].sum()
df_tv_grouped.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * OOH dataset

# COMMAND ----------

df_ooh = pd.read_csv(ooh_file)
dict_ooh_rename_cols = {'Date':'ooh_date', 'Total Cost':'ooh_cost'}
df_ooh.rename(columns=dict_ooh_rename_cols, inplace=True)
df_ooh.columns = df_ooh.columns.str.lower()
df_ooh['ooh_date'] = pd.to_datetime(df_ooh['ooh_date'])
df_ooh_grouped = df_ooh.groupby('ooh_date', as_index=False)['ooh_cost'].sum()
df_ooh_grouped.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * External features datasets

# COMMAND ----------

df_ext_cpi = pd.read_excel(ext_file,
                           engine='openpyxl',
                           sheet_name='UK CPI Data')
df_ext_unemplmnt = pd.read_excel(ext_file,
                           engine='openpyxl',
                           sheet_name='UK Unemployment Data')
df_ext_covid = pd.read_excel(ext_file,
                           engine='openpyxl',
                           sheet_name='UK Covid Cases')
df_ext_pubhol = pd.read_excel(ext_file,
                           engine='openpyxl',
                           sheet_name='UK Public Holidays')

# COMMAND ----------

# Make a single dataset for external dataset

df_ext_cpi_unemplmnt = pd.merge(df_ext_cpi, df_ext_unemplmnt, on='year_month', how='outer', indicator=True)
df_ext_cpi_unemplmnt['first_merge'] = df_ext_cpi_unemplmnt['_merge']
df_ext_cpi_unemplmnt.drop(columns=['_merge'], axis=1, inplace=True)
df_ext_cpi_unemplmnt_covid = pd.merge(df_ext_cpi_unemplmnt, df_ext_covid, left_on='year_month', right_on='date', how='outer', indicator=True)
df_ext_cpi_unemplmnt_covid['second_merge'] = df_ext_cpi_unemplmnt_covid['_merge']
df_ext_cpi_unemplmnt_covid.drop(columns=['_merge'], axis=1, inplace=True)
df_ext_cpi_unemplmnt_covid_pubhol = pd.merge(df_ext_cpi_unemplmnt_covid, df_ext_pubhol, left_on='year_month', right_on='Date', how='outer', indicator=True)
df_ext_cpi_unemplmnt_covid_pubhol['third_merge'] = df_ext_cpi_unemplmnt_covid_pubhol['_merge']
df_ext_cpi_unemplmnt_covid_pubhol.drop(columns=['_merge'], axis=1, inplace=True)

# COMMAND ----------

# Select relevant columns from external datasets

ext_rel_cols = ['year_month', 'CPIH', 'CPI', 'OOH', 'unemployment_percentage',
                                            'newCasesByPublishDate', 'Type']
df_ext = df_ext_cpi_unemplmnt_covid_pubhol[ext_rel_cols]
dict_ext_rename_cols = {'year_month':'ext_date', 'unemployment_percentage':'unemplmnt_prct',
                      'newCasesByPublishDate':'new_covid_case', 'Type':'pub_hol'}                                            
df_ext.rename(columns=dict_ext_rename_cols, inplace=True)
df_ext.columns = df_ext.columns.str.lower()
df_ext['ext_date'] = pd.to_datetime(df_ext['ext_date'])
df_ext.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC * Attributed channels dataset

# COMMAND ----------

df_snf = pd.read_csv(snf_file)
dict_snf_rename_cols = {'Partnerships & Referrals':'Partnerships_Referrals', 'Paid Search':'Paid_Search',
                      'Organic Search':'Organic_Search', 'Apple Search Ads':'Apple_Search_Ads', 'Social Paid':'Social_Paid'}
df_snf.rename(columns=dict_snf_rename_cols,
              inplace=True)
df_snf.columns = df_snf.columns.str.lower()
df_snf['km_date'] = pd.to_datetime(df_snf['km_date'])
df_snf.head(3)

# COMMAND ----------

df_crm = pd.read_csv(crm_file)
dict_crm_rename_cols = {'BATCH_DATE':'crm_date', 'COUNT(CAMPAIGN_ID)':'crm_email'}
df_crm.rename(columns=dict_crm_rename_cols, inplace=True)
df_crm['crm_date'] = pd.to_datetime(df_crm['crm_date'])
df_crm.head(2)

# COMMAND ----------

df_partnerships = pd.read_csv(prtnrshps_file)
dict_prtnrshps_rename_cols = {'spend':'partnerships'}
df_partnerships.rename(columns=dict_prtnrshps_rename_cols, inplace=True)
df_partnerships['partnerships_date'] = pd.to_datetime(df_partnerships['partnerships_date'])
df_partnerships.head(3)

# COMMAND ----------

df_partnerships_incentive = pd.read_csv(prtnrshps_incntv_file)
df_partnerships_incentive['km_date'] = pd.to_datetime(df_partnerships_incentive['km_date'])
df_partnerships_incentive.head(3)

# COMMAND ----------

df_affiliates = pd.read_csv(afflts_file)
dict_afflts_rename_cols = {'affiliate_date':'km_date', 'affiliate_cost':'affiliate'}
df_affiliates['affiliate_date'] = pd.to_datetime(df_affiliates['affiliate_date'])
df_affiliates['affiliate_date'] = pd.to_datetime(df_affiliates['affiliate_date'].dt.date)
df_affiliates.rename(columns = dict_afflts_rename_cols, inplace = True)
df_affiliates.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. <a href='#MakeaSingleDataframe'>Make a Single Dataframe</a>

# COMMAND ----------

# # compile the list of dataframes for merge 
# # (Due to need for sanity of date column names, not using shortcuts, rather merging individually)

# data_frames_list = [df_snf, df_tv_grouped, df_ooh_grouped, df_ext, df_crm, df_partnerships, df_partnerships_incentive, df_affiliates]

# df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['km_date'], how='outer'), data_frames_list)

# COMMAND ----------

# Merge all the datasets individually

df_snf_tv = pd.merge(df_snf, df_tv_grouped, left_on='km_date', right_on='tv_date', how='left', indicator=True)
df_snf_tv['first_merge'] = df_snf_tv['_merge']
df_snf_tv.drop(columns=['_merge'], axis=1, inplace=True)
df_snf_tv_ooh = pd.merge(df_snf_tv, df_ooh_grouped, left_on='km_date', right_on='ooh_date', how='left', indicator=True)
df_snf_tv_ooh['second_merge'] = df_snf_tv_ooh['_merge']
df_snf_tv_ooh.drop(columns=['_merge'], axis=1, inplace=True)
df_snf_tv_ooh_ext = pd.merge(df_snf_tv_ooh, df_ext, left_on='km_date', right_on='ext_date', how='left', indicator=True)
df_snf_tv_ooh_ext['third_merge'] = df_snf_tv_ooh_ext['_merge']
df_snf_tv_ooh_ext.drop(columns=['_merge'], axis=1, inplace=True)
df_snf_tv_ooh_ext_crm = pd.merge(df_snf_tv_ooh_ext, df_crm, left_on='km_date', right_on='crm_date', how='left', indicator=True)
df_snf_tv_ooh_ext_crm['fourth_merge'] = df_snf_tv_ooh_ext_crm['_merge']
df_snf_tv_ooh_ext_crm.drop(columns=['_merge'], axis=1, inplace=True)
df_snf_tv_ooh_ext_crm_prtnrshp = pd.merge(df_snf_tv_ooh_ext_crm, df_partnerships, left_on='km_date', right_on='partnerships_date', how='left', indicator=True)
df_snf_tv_ooh_ext_crm_prtnrshp['fifth_merge'] = df_snf_tv_ooh_ext_crm_prtnrshp['_merge']
df_snf_tv_ooh_ext_crm_prtnrshp.drop(columns=['_merge'], axis=1, inplace=True)
df_snf_tv_ooh_ext_crm_prtnrshp_incntv = pd.merge(df_snf_tv_ooh_ext_crm_prtnrshp, df_partnerships_incentive, left_on='km_date', right_on='km_date', how='left', indicator=True)
df_snf_tv_ooh_ext_crm_prtnrshp_incntv['sixth_merge'] = df_snf_tv_ooh_ext_crm_prtnrshp_incntv['_merge']
df_snf_tv_ooh_ext_crm_prtnrshp_incntv.drop(columns=['_merge'], axis=1, inplace=True)
df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates = pd.merge(df_snf_tv_ooh_ext_crm_prtnrshp_incntv, df_affiliates, left_on='km_date', right_on='km_date', how='left', indicator=True)
df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates['seventh_merge'] = df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates['_merge']
df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates.drop(columns=['_merge'], axis=1, inplace=True)

# COMMAND ----------

df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates.drop(columns=['ooh'], axis=1, inplace=True)
df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates['partnerships'] = df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates['partnerships'] + df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates['partnership_incentive']
merged_df_rename_cols = {'tv_cost':'tv', 'ooh_cost':'ooh', 'affiliate_y':'affiliates'}
df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates.rename(columns=merged_df_rename_cols, inplace=True)

# COMMAND ----------

df_offline_attrbtn = df_snf_tv_ooh_ext_crm_prtnrshp_incntv_affiliates[['km_date', 'km_count', 'paid_search',
                                        'display', 'uac', 'apple_search_ads', 'social_paid', 'tv', 'ooh',
                                        'partnerships', 'affiliates', 'crm_email',
                                        'cpi', 'unemplmnt_prct', 'pub_hol']]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create week-level aggregated dataset

# COMMAND ----------

df_attrib_weekly = df_offline_attrbtn.copy()
df_attrib_weekly.index = pd.to_datetime(df_attrib_weekly.km_date)
df_attrib_weekly.drop(columns=['km_date'], axis=1, inplace=True)
df_attrib_weekly = df_attrib_weekly.resample('W-Sun').sum()
df_attrib_weekly = df_attrib_weekly.reset_index(level=0)
df_attrib_weekly['km_date_start'] = df_attrib_weekly['km_date'] -  pd.to_timedelta(6, unit='d')
df_attrib_weekly['km_date'] = df_attrib_weekly['km_date_start']
df_attrib_weekly.drop(columns=['km_date_start'], axis=1, inplace=True)
df_attrib_weekly = df_attrib_weekly[(df_attrib_weekly['km_date']>='2022-01-01')&(df_attrib_weekly['km_date']<='2023-03-31')]

# COMMAND ----------

dataset = df_attrib_weekly.copy()

# COMMAND ----------

# # #rows and #features
# print("#rows: ", dataset.shape[0], "\n"
#       "#features: ", dataset.shape[1])
# dataset.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Export Data into a csv file<a id="ExportDataintoacsvfile"></a>

# COMMAND ----------

# offline_attrbtn_data_out_dir = file_location
# offline_attrbtn_data_out_file = 'offline_attribution_train_test_data.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Run below cell only when new data generation is required

# COMMAND ----------

# # Load dataframe as csv to DBFS

# def load_dataframe_to_dbfs(out_dir, out_file, dataframe):
#   dataframe.to_csv(out_dir+out_file, index=False, encoding="utf-8")
  
# load_dataframe_to_dbfs(offline_attrbtn_data_out_dir, offline_attrbtn_data_out_file, dataset)

# COMMAND ----------

# def log_train_test_raw_data(exp_run_name, exp_id, train_test_data_file):
#   '''
#   This function logs the raw data
#   '''
#   with mlflow.start_run(run_name=exp_run_name, experiment_id=exp_id) as run:
#     # Log Artifacts
#     mlflow.log_artifact(train_test_data_file)
      
#   mlflow.end_run()
  
# # Log raw data to mlflow experiments
# log_train_test_raw_data("offline_attribution_save_raw_data", ex_id, offline_attrbtn_data_out_dir+offline_attrbtn_data_out_file)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Below cells are for review and convenience of data check

# COMMAND ----------

# Only for development phase

df_offline_attrbtn_train_test_data = pd.read_csv(offline_attrbtn_data_out_dir+offline_attrbtn_data_out_file)
# #rows and #features
print("#rows: ", df_offline_attrbtn_train_test_data.shape[0], "\n"
      "#columns: ", df_offline_attrbtn_train_test_data.shape[1])
df_offline_attrbtn_train_test_data.head(1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training and Test Data Consideration Period

# COMMAND ----------

print("Train Test data consideration lower range: ", (pd.to_datetime(df_offline_attrbtn_train_test_data["km_date"]).min()).strftime('%Y-%m-%d'))
print("Train Test data consideration upper range: ", (pd.to_datetime(df_offline_attrbtn_train_test_data["km_date"]).max()).strftime('%Y-%m-%d'))

# COMMAND ----------

# MAGIC %md
# MAGIC * End of Notebook
