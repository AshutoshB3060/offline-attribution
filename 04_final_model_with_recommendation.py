# Databricks notebook source
# MAGIC %md 
# MAGIC ## Objective:
# MAGIC * This notebook contains code for training model with all the data points and provides recommendation for optimum budget allocation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC ### 1. <a href='#InstallDependencies'>Install Dependencies</a>
# MAGIC ### 2. <a href='#ImportData'>Import Data</a>
# MAGIC ### 3. <a href='#TrainTestDataPreparation'>Training Data Preparation</a>
# MAGIC ### 4. <a href='#ModelBuild'>Model Build</a>
# MAGIC ### 5. <a href='#Recommendation'>Recommendation</a>

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Install Dependencies <a id='InstallDependencies'></a>

# COMMAND ----------

# dbutils.library.installPyPI('hyperopt', version='0.2.7')
# dbutils.library.installPyPI('mlflow', version='2.2.2')
# !pip install lightweight-mmm==0.1.5
# # !pip uninstall matplotlib
# # !pip install matplotlib==3.2.1
# dbutils.library.restartPython()

# COMMAND ----------

# # widgets from dbutils

# dbutils.widgets.removeAll()
# dbutils.widgets.text("train_test_data_file", "/dbfs/FileStore/FileStore/offline_attribution/datasets/offline_attribution_train_test_data.csv")
# train_test_data_file = dbutils.widgets.getArgument(name='train_test_data_file')
# dbutils.widgets.text("experiment_path", "/Shared/Analytics/offline_attribution/offline_attribution_experiments")
# experiment_path = dbutils.widgets.getArgument(name='experiment_path')
# dbutils.widgets.text("visualisation_plot_path", "/dbfs/FileStore/FileStore/offline_attribution/datasets/visualisation_plot/")
# visualisation_plot_path = dbutils.widgets.getArgument(name='visualisation_plot_path')
# dbutils.widgets.text("file_location", "/dbfs/FileStore/FileStore/offline_attribution/datasets/")
# file_location = dbutils.widgets.getArgument(name='file_location')

# COMMAND ----------

# Handle warnings (during execution of code)
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# Import Libraries

import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
from datetime import datetime
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils
%matplotlib inline
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pprint
pp = pprint.PrettyPrinter(indent=4)
import mlflow
import mlflow.pyfunc
from  mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

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

# Declare variables

#reproducibility
SEED = 12

# To run chains in parallel
numpyro.set_host_device_count(3)

# preprocessing, model parameter
model_type = "carryover"
cost_multiplier = 0.15
num_week_recommend = 4

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Import Data<a id="ImportData"></a>  

# COMMAND ----------

# Only for development phase

df_offline_attrbtn_train = pd.read_csv(train_test_data_file)

# #rows and #features
print("#rows: ", df_offline_attrbtn_train.shape[0], "\n"
      "#columns: ", df_offline_attrbtn_train.shape[1])
df_offline_attrbtn_train.head(3)

# COMMAND ----------

# Check the date range for the dataset

print(df_offline_attrbtn_train['km_date'].min())
print(df_offline_attrbtn_train['km_date'].max())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Training Data Preparation<a id="TrainingDataPreparation"></a>  

# COMMAND ----------

# Create a copy of the dataset for train and test

df_train_data_prep = df_offline_attrbtn_train.copy()

# COMMAND ----------

# Define datasets with media variables, external features and target feature

media_spend = ['paid_search', 'display', 'uac', 'apple_search_ads',
                              'social_paid', 'tv', 'ooh', 'partnerships',
                              'affiliates']

df_media_data = df_train_data_prep[media_spend]

df_extra_features = df_train_data_prep[['cpi', 'unemplmnt_prct', 'pub_hol']]

df_target = df_train_data_prep['km_count']

df_costs = df_train_data_prep[media_spend].sum(axis=0)

# COMMAND ----------

# Preprocess the data for model consumption
np_media_data = df_media_data.to_numpy()
media_data = jnp.array(np_media_data)

np_extra_features = df_extra_features.to_numpy()
extra_features = jnp.array(np_extra_features)

np_target = df_target.to_numpy()
target = jnp.array(np_target)

np_costs = df_costs.values
costs = jnp.array(np_costs)

# COMMAND ----------

# Training sample

'''
There are 65 weeks of data available.
We can use 61 weeks of data for training and 4 weeks for recommendation
'''

data_size = 65
split_point = data_size - num_week_recommend

media_data_train = media_data[:split_point, ...]
extra_features_train = extra_features[:split_point, ...]
extra_features_test = extra_features[split_point:, ...]
target_train = target[:split_point]

# COMMAND ----------

# Scaling of features

# Define scaler
media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=cost_multiplier)

# Transform training datasets
media_data_train = media_scaler.fit_transform(media_data_train)
extra_features_train = extra_features_scaler.fit_transform(extra_features_train)
target_train = target_scaler.fit_transform(target_train)
costs = cost_scaler.fit_transform(costs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Model Train Test<a id="ModelTrainTest"></a>  

# COMMAND ----------

# Define the baseline model
mmm = lightweight_mmm.LightweightMMM(model_name=model_type)

# COMMAND ----------

# Fit the model with best parameters

number_warmup=1200
number_samples=1200

mmm.fit(
    media=media_data_train,
    media_prior=costs,
    target=target_train,
    extra_features=extra_features_train,
    number_warmup=number_warmup,
    number_samples=number_samples,
    seed=SEED)

# COMMAND ----------

# Save the model to DBFS path

file_name = "offline_attrbtn_media_mix_model.pkl"
utils.save_model(media_mix_model=mmm, file_path=file_location+file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualisation of media spend and response distribution

# COMMAND ----------

# MAGIC %md
# MAGIC * Baseline and media channel attribution over time

# COMMAND ----------

media_channel_list = ('paid_search', 'display', 'uac', 'apple_search_ads',
                      'social_paid', 'tv_cost', 'ooh_cost', 'partnerships', 'affiliates')

# COMMAND ----------

plot.plot_media_baseline_contribution_area_plot(media_mix_model=mmm,
                                                target_scaler=target_scaler,
                                                channel_names=media_channel_list,
                                                fig_size=(30,10))

# COMMAND ----------

# MAGIC %md
# MAGIC * Media contribution percentage and ROI

# COMMAND ----------

media_contribution, roi_hat = mmm.get_posterior_metrics(target_scaler=target_scaler, cost_scaler=cost_scaler)

# COMMAND ----------

plot.plot_bars_media_metrics(metric=media_contribution, metric_name="Media Contribution Percentage",
                             channel_names=media_channel_list)

# COMMAND ----------

plot.plot_bars_media_metrics(metric=roi_hat, metric_name="Return on Investment",
                             channel_names=media_channel_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Budget Allocation Optimisation

# COMMAND ----------

# Define variables

prices = jnp.ones(mmm.n_media_channels)
n_time_periods = num_week_recommend
budget = jnp.sum(jnp.dot(prices, media_data.mean(axis=0)))* n_time_periods

# COMMAND ----------

# Run optimization

solution, kpi_without_optim, previous_budget_allocation = optimize_media.find_optimal_budgets(
    n_time_periods=n_time_periods,
    media_mix_model=mmm,
    extra_features=extra_features_scaler.transform(extra_features_test)[:n_time_periods],
    budget=budget,
    prices=prices,
    media_scaler=media_scaler,
    target_scaler=target_scaler,
    seed=SEED)

# COMMAND ----------

# Obtain the optimal weekly allocation
optimal_buget_allocation = prices * solution.x

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Recommendation<a id="Recommendation"></a>  

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Pre and Post Optimisation Budget Allocation

# COMMAND ----------

# Plot out pre post optimization budget allocation and predicted target variable comparison
plot.plot_pre_post_budget_allocation_comparison(media_mix_model=mmm, 
                                                kpi_with_optim=solution['fun'], 
                                                kpi_without_optim=kpi_without_optim,
                                                optimal_buget_allocation=optimal_buget_allocation, 
                                                previous_budget_allocation=previous_budget_allocation,
                                                channel_names=media_channel_list,
                                                figure_size=(10,10))

# COMMAND ----------

# MAGIC %md
# MAGIC This model recommends to change the budget allocation strategy to see an uplift in KMs in 4 weeks
# MAGIC Model recommendation is directional. We should consider other marketing channels strategies too which were not included in modelling due to various reasons.

# COMMAND ----------

# MAGIC %md
# MAGIC * Notes for model consumption
# MAGIC 
# MAGIC 1. For accurate recommendation, re-run the model every time new data is received. This will accomodate the recent marketing budget changes.
# MAGIC 2. Model is not logged into Mlflow registry due to above reason. For reference, a pickle file is saved in DBFS path.

# COMMAND ----------

# MAGIC %md
# MAGIC * End of this Notebook
