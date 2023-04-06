# Databricks notebook source
# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC ### 1. <a href='#InstallDependencies'>Install Dependencies</a>
# MAGIC ### 2. <a href='#ImportData'>Import Data</a>
# MAGIC ### 3. <a href='#EDA'>EDA</a>

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Install Dependencies <a id='InstallDependencies'></a>

# COMMAND ----------

# dbutils.library.installPyPI('mlflow', version='2.2.2')
# dbutils.library.installPyPI('plotly', version='5.1.0')
# dbutils.library.restartPython()

# COMMAND ----------

# # widgets from dbutils

# dbutils.widgets.removeAll()
# dbutils.widgets.text("visualisation_plot_path", "/dbfs/FileStore/FileStore/offline_attribution/datasets/visualisation_plot/")
# visualisation_plot_path = dbutils.widgets.getArgument(name='visualisation_plot_path')
# dbutils.widgets.text("train_test_data_file", "/dbfs/FileStore/FileStore/offline_attribution/datasets/offline_attribution_train_test_data.csv")
# train_test_data_file = dbutils.widgets.getArgument(name='train_test_data_file')
# dbutils.widgets.text("experiment_path", "/Shared/Analytics/offline_attribution/offline_attribution_experiments")
# experiment_path = dbutils.widgets.getArgument(name='experiment_path')

# COMMAND ----------

# Handle warnings (during execution of code)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# COMMAND ----------

# Import Libraries

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
import pprint
pp = pprint.PrettyPrinter(indent=4)
import mlflow
from  mlflow.tracking import MlflowClient
# from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

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
# MAGIC ### Utility Functions

# COMMAND ----------

def null_perc_check (df):
    '''
    Calculates missing value count and percentage for all the columns in a dataframe

    Inputs
    -------
    df : dataframe
        The dataframe for which missing value distribution needs to checked

    Output
    -------
    dataframe
        a dataframe showing missing value count and percentage for all the columns
    '''
    missing_value_df = pd.DataFrame(index = df.keys(), data =df.isnull().sum(), columns = ['Missing_Value_Count'])
    missing_value_df['Missing_Value_Percentage'] = np.round(((df.isnull().mean())*100),1)
    sorted_df = missing_value_df.sort_values('Missing_Value_Count',ascending= False)
    return sorted_df

def ts_line_chart (df, x_val, y_val, title):
  '''
    Plots time-series line chart

    Inputs
    -------
    df : dataframe
        The dataframe for which missing value distribution needs to checked
    x_val : string
        Value for x-axis
    y_val : string
        Value for y-axis
    title : string
        Title of the chart

    Output
    -------
    Line chart
    '''
  fig = px.line(df_ts_line, x=x_val, y=y_val,
              hover_data={x_val: "|%d %B %Y"},
              title=title)
  fig.update_xaxes(dtick="M1",
                  tickformat="%b\n%Y")
  fig.show()



def ts_stacked_bar_chart (df, x_val, y_val, title):
  '''
    Plots time-series bar chart

    Inputs
    -------
    df : dataframe
        The dataframe for which missing value distribution needs to checked
    x_val : string
        Value for x-axis
    y_val : string
        Value for y-axis
    title : string
        Title of the chart

    Output
    -------
    Stacked bar chart
    '''
  fig = px.bar(df, x=x_val, y=y_val, title=title)
  fig.update_layout(yaxis={'categoryorder':'total ascending'})
  fig.show()


def log_eda_plots(exp_run_name, exp_id, vis_plot_path, vis_plot_list):
  '''
  This function logs the plots generated during EDA
  '''
  with mlflow.start_run(run_name=exp_run_name, experiment_id=exp_id) as run:
    # Log Artifacts
    for plot in vis_plot_list:
      mlflow.log_artifact(vis_plot_path + plot)

  mlflow.end_run()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Import Data<a id="ImportData"></a>  

# COMMAND ----------

# Only for development phase

df_train_test_offline_attrbtn = pd.read_csv(train_test_data_file)
# #rows and #features
print("#rows: ", df_train_test_offline_attrbtn.shape[0], "\n"
      "#features: ", df_train_test_offline_attrbtn.shape[1])
df_train_test_offline_attrbtn.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Exploratory Data Analysis<a id="ExploratoryDataAnalysis"></a>  

# COMMAND ----------

# Create a copy of the dataset imported for basic understanding and analysis
df_offline_attrbtn_eda = df_train_test_offline_attrbtn.copy()
print(df_offline_attrbtn_eda.shape)

# COMMAND ----------

# Missing Value Statistics

null_perc_check(df_offline_attrbtn_eda)

# COMMAND ----------

# MAGIC %md
# MAGIC * There are no missing values in the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA Plan
# MAGIC 
# MAGIC * Budget Spend
# MAGIC 1. Time series line chart   
# MAGIC 2. Time series stacked-bar chart
# MAGIC 3. Time series percentage stacked-bar chart (Distribution with TV & OOH)
# MAGIC 4. Time series percentage stacked-bar chart (Distribution without TV & OOH)
# MAGIC 
# MAGIC * KM Distribution
# MAGIC 1. Weekly KM distribution

# COMMAND ----------

# MAGIC %md
# MAGIC #### Channel-wise weekly budget spend

# COMMAND ----------

channels = ['paid_search', 'display', 'uac',
       'apple_search_ads', 'social_paid', 'tv', 'ooh', 'partnerships',
       'affiliates']

# COMMAND ----------

# Time series line chart

df_ts_line = df_offline_attrbtn_eda.copy()

ts_line_chart(df_ts_line, "km_date", channels, 'Channel-wise weekly budget spend')
plt.savefig(visualisation_plot_path + "offline_attrbtn_eda_weekly_spend_line_chart.png")

# COMMAND ----------

# Time series stacked bar chart

df_ts_stackbar = df_offline_attrbtn_eda.copy()

ts_stacked_bar_chart(df_ts_stackbar, "km_date", channels, 'Channel-wise weekly budget actual spend distribution')
plt.savefig(visualisation_plot_path + "offline_attrbtn_eda_weekly_spend_stacked_bar_chart.png")

# COMMAND ----------

# Time series percentage stacked bar chart

df_ts_perc_stackbar = df_offline_attrbtn_eda.copy()

df_ts_perc_stackbar['total_spend_act'] = df_ts_perc_stackbar[channels].sum(axis=1)

for col in channels:
  df_ts_perc_stackbar[col+'_perc'] = np.round((df_ts_perc_stackbar[col]/df_ts_perc_stackbar['total_spend_act'])*100,2)

df_ts_perc_stackbar.drop(columns=channels, axis=1, inplace=True)

channels_perc_stack = ['paid_search_perc', 'display_perc', 'uac_perc',
       'apple_search_ads_perc', 'social_paid_perc', 'tv_perc', 'ooh_perc', 'partnerships_perc',
       'affiliates_perc']

ts_stacked_bar_chart(df_ts_perc_stackbar, "km_date", channels_perc_stack, 'Channel-wise weekly budget spend percentage distribution')
plt.savefig(visualisation_plot_path + "offline_attrbtn_eda_weekly_spend_percentage_distribution.png")

# COMMAND ----------

# Time series percentage stacked bar chart (without TV & OOH)

df_ts_perc_stackbar_without_tvooh = df_offline_attrbtn_eda.copy()

channels_without_tvooh = ['paid_search', 'display', 'uac',
       'apple_search_ads', 'social_paid', 'partnerships',
       'affiliates']

df_ts_perc_stackbar_without_tvooh['total_spend_act'] = df_ts_perc_stackbar_without_tvooh[channels_without_tvooh].sum(axis=1)

for col in channels_without_tvooh:
  df_ts_perc_stackbar_without_tvooh[col+'_perc'] = np.round((df_ts_perc_stackbar_without_tvooh[col]/df_ts_perc_stackbar_without_tvooh['total_spend_act'])*100,2)
  
df_ts_perc_stackbar_without_tvooh.drop(columns=channels_without_tvooh, axis=1, inplace=True)

channels_perc_stack_without_tvooh = ['paid_search_perc', 'display_perc', 'uac_perc',
       'apple_search_ads_perc', 'social_paid_perc', 'partnerships_perc',
       'affiliates_perc']

ts_stacked_bar_chart(df_ts_perc_stackbar_without_tvooh, "km_date", channels_perc_stack_without_tvooh, 'Channel-wise weekly budget spend percentage distribution (Without TV & OOH)')
plt.savefig(visualisation_plot_path + "offline_attrbtn_eda_weekly_spend_percentage_distribution_without_tvooh.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weekly KM Distribution 

# COMMAND ----------

# Time series line chart for weekly KM distribution

df_ts_line_km = df_offline_attrbtn_eda.copy()

ts_line_chart(df_ts_line_km, "km_date", "km_count", 'Weekly KM distribution')
plt.savefig(visualisation_plot_path + "offline_attrbtn_eda_weekly_km_distribution.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Log the artifacts in Mlflow experiments

# COMMAND ----------

vis_plot_list = ["offline_attrbtn_eda_weekly_spend_line_chart.png", "offline_attrbtn_eda_weekly_spend_stacked_bar_chart.png",
                "offline_attrbtn_eda_weekly_spend_percentage_distribution.png", "offline_attrbtn_eda_weekly_spend_percentage_distribution_without_tvooh.png",
                "offline_attrbtn_eda_weekly_km_distribution.png"]

# COMMAND ----------

# Log EDA plots
log_eda_plots("offline_attribution_eda", ex_id, visualisation_plot_path, vis_plot_list)

# COMMAND ----------

# MAGIC %md
# MAGIC * End of Notebook
