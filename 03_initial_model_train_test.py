# Databricks notebook source
# MAGIC %md 
# MAGIC ## Objective:
# MAGIC * This notebook contains code for model training and test.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC ### 1. <a href='#InstallDependencies'>Install Dependencies</a>
# MAGIC ### 2. <a href='#ImportData'>Import Data</a>
# MAGIC ### 3. <a href='#TrainTestDataPreparation'>Train Test Data Preparation</a>
# MAGIC ### 4. <a href='#ModelTrainTest'>Model Train Test</a>

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
from hyperopt import fmin, tpe,rand, hp, STATUS_OK,space_eval, SparkTrials

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Import Data<a id="ImportData"></a>  

# COMMAND ----------

# Only for development phase

df_offline_attrbtn_train_test = pd.read_csv(train_test_data_file)

# #rows and #features
print("#rows: ", df_offline_attrbtn_train_test.shape[0], "\n"
      "#columns: ", df_offline_attrbtn_train_test.shape[1])
df_offline_attrbtn_train_test.head(3)

# COMMAND ----------

# Check the date range for the dataset

print(df_offline_attrbtn_train_test['km_date'].min())
print(df_offline_attrbtn_train_test['km_date'].max())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Train Test Data Preparation<a id="TrainTestDataPreparation"></a>  

# COMMAND ----------

# Create a copy of the dataset for train and test

df_train_test_data_prep = df_offline_attrbtn_train_test.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Steps invloved in data prep and model build and evaluation
# MAGIC 1. data preprocessing
# MAGIC 2. define model
# MAGIC 3. optimise 

# COMMAND ----------

# Define datasets with media variables, external features and target feature

media_spend = ['paid_search', 'display', 'uac', 'apple_search_ads',
                              'social_paid', 'tv', 'ooh', 'partnerships',
                              'affiliates']

df_media_data = df_train_test_data_prep[media_spend]

df_extra_features = df_train_test_data_prep[['cpi', 'unemplmnt_prct', 'pub_hol']]

df_target = df_train_test_data_prep['km_count']

df_costs = df_train_test_data_prep[media_spend].sum(axis=0)

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

# Train-test split

'''
There are 65 weeks of data available.
Training period : 1st 57 weeks
Test period : last 8 weeks
'''

data_size = 65

# Split and scale data.
split_point = data_size - 8

# Media data
media_data_train = media_data[:split_point, ...]
media_data_test = media_data[split_point:, ...]
# Extra features
extra_features_train = extra_features[:split_point, ...]
extra_features_test = extra_features[split_point:, ...]
# Target
target_train = target[:split_point]
target_test = target[split_point:]

# COMMAND ----------

extra_features_test

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

# Transform test datasets
media_data_test = media_scaler.fit_transform(media_data_test)
extra_features_test = extra_features_scaler.fit_transform(extra_features_test)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Model Train Test<a id="ModelTrainTest"></a>  

# COMMAND ----------

# Define the baseline model
mmm = lightweight_mmm.LightweightMMM(model_name=model_type)

# COMMAND ----------

# Baseline model fitting with training data

mmm.fit(
    media=media_data_train,
    media_prior=costs,
    target=target_train,
    extra_features=extra_features_train,
    number_warmup=100,
    number_samples=100,
    seed=SEED)

# COMMAND ----------

# Baseline model prediction for future weeks

baseline_predictions_future_8wks = mmm.predict(media=media_data_test,
                              extra_features=extra_features_test,
                              seed=SEED)

# COMMAND ----------

# Baseline model prediction (Including all data points)

baseline_predictions_all_data = mmm.predict(media=media_data,
                              extra_features=extra_features,
                              seed=SEED)

# COMMAND ----------

plot.plot_out_of_sample_model_fit(out_of_sample_predictions=baseline_predictions_future_8wks,
                                 out_of_sample_target=target_test)

# COMMAND ----------

plot.plot_out_of_sample_model_fit(out_of_sample_predictions=baseline_predictions_all_data,
                                 out_of_sample_target=target_scaler.transform(target))

# COMMAND ----------

# MAGIC %md
# MAGIC * Baseline MAPE for hold-out future 8 weeks : 
# MAGIC * Baseline MAPE for all-data including future 8 weeks : 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Tune Hyperparameters

# COMMAND ----------

# MAGIC %md
# MAGIC * Uncomment below 2 cells for tuning hyperparameters

# COMMAND ----------

# # # Define all the parameters

# param_iterations = list(np.arange(100, 1500, 100))
# param_samples = list(np.arange(100, 1500, 100))
# param_max_iterations = list(np.arange(100, 500, 50))
# param_bounds_pct = list(np.arange(0.05, 0.25, 0.05))
# param_model_type = ['carryover', 'adstock', 'hill_adstock']

# COMMAND ----------

# def gen_best_params(media_train, extra_ftrs_train, target_train, media_test, extra_ftrs_test, target_test, costs, model_type):
   
  # def mape(y_test, pred):
  #   y_test, pred = np.array(y_test), np.array(pred)
  #   mape = np.mean(np.abs((y_test - pred) / y_test))
  #   return mape

  # def objective(params):
  #     #define model
  #     model = lightweight_mmm.LightweightMMM(model_name=model_type)
  #     # train model
  #     mmm.fit(
  #       media=media_train,
  #       media_prior=costs,
  #       target=target_train,
  #       extra_features=extra_ftrs_train,
  #       number_warmup=num_warmup,
  #       number_samples=num_samples,
  #       seed=random_seed)
  #     # make predictions
  #     predictions = model.predict(media=media_test,
  #                             extra_features=extra_ftrs_test,
  #                             seed=random_seed)
  #     # obtain mape
  #     mape = mape(target_test, predictions)
  #     # invert metric for hyperopt
  #     loss = mape

  #     # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
  #     return {'loss': loss, 'status': STATUS_OK}

#   search_space = {
#     'num_warmup':hp.choice('num_warmup', param_iterations),
#     'num_samples':hp.choice('num_samples', param_samples),
#     'max_iterations':hp.choice('max_iterations', param_max_iterations),
#     'bounds_lower_pct':hp.choice('bounds_lower_pct', param_bounds_pct),
#     'bounds_upper_pct':hp.choice('bounds_upper_pct', param_bounds_pct),
#     'model_name':hp.choice('model_name', param_model_type),
#     'random_state': hp.choice('random_state', [SEED])
#   }
    
#   argmin = fmin(
#     fn=objective,
#     space=search_space,
#     algo=tpe.suggest, 
#     max_evals=20,
#     trials=SparkTrials(parallelism=16),
#     verbose=True
#   )

#   best_params = space_eval(search_space, argmin)
#   df_best_params = pd.DataFrame.from_dict(best_params, orient='index', columns=['value'])
#   return(best_params)

# COMMAND ----------

best_params = {
  'num_warmup': 1200,
  'num_samples': 1000,
  'cost_multiplier': 0.15,
  'max_iterations': 400,
  'bounds_lower_pct': 0.2,
  'bounds_upper_pct': 0.2,
  'model_name':'carryover'
}


# COMMAND ----------

# MAGIC %md
# MAGIC #### Cross-Validation

# COMMAND ----------

# MAGIC %md
# MAGIC * Cross-validation with random samples can't be performed, due to requirement of data points in sequence.
# MAGIC * Rather manually, performance is checked with lesser number of weeks of datapoints.
# MAGIC * When number of data points is less, error (MAPE) increases.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Testing strategy (Out-of-time validation)
# MAGIC 1. Train on 57 weeks of data, test on 4 weeks
# MAGIC 2. Train on 61 weeks of data, test on 4 weeks
# MAGIC 2. Train on 57 weeks of data, test on 8 weeks

# COMMAND ----------

# MAGIC %md
# MAGIC * Backtesting is not applicable as the model predicts the media response distribution for future weeks

# COMMAND ----------

data_size = 65

# Split and scale data.
split_point_61wks = data_size - 4

# Media data
media_data_train_61wks = media_data[:split_point_61wks, ...]
media_data_test_61wks = media_data[split_point_61wks:, ...]
# Extra features
extra_features_train_61wks = extra_features[:split_point_61wks, ...]
extra_features_test_61wks = extra_features[split_point_61wks:, ...]
# Target
target_train_61wks = target[:split_point_61wks]
target_test_61wks = target[split_point_61wks:]

# COMMAND ----------

# Scaling of features

# Transform training datasets
media_data_train_61wks = media_scaler.fit_transform(media_data_train_61wks)
extra_features_train_61wks = extra_features_scaler.fit_transform(extra_features_train_61wks)
target_train_61wks = target_scaler.fit_transform(target_train_61wks)

# Transform test datasets
media_data_test_61wks = media_scaler.fit_transform(media_data_test_61wks)
extra_features_test_61wks = extra_features_scaler.fit_transform(extra_features_test_61wks)

# COMMAND ----------

mmm.fit(
    media=media_data_train,
    media_prior=costs,
    target=target_train,
    extra_features=extra_features_train,
    number_warmup=1200,
    number_samples=1000,
    seed=SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Testing Strategy 1: 57 weeks training and 4 weeks predictions

# COMMAND ----------

preds_train_57wks_test_4wks = mmm.predict(media=media_data_test_61wks,
                              extra_features=extra_features_test_61wks,
                              seed=SEED)

# COMMAND ----------

plot.plot_out_of_sample_model_fit(out_of_sample_predictions=preds_train_57wks_test_4wks,
                                 out_of_sample_target=target_test_61wks)

# COMMAND ----------

# MAGIC %md
# MAGIC MAPE : 7.9

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Testing Strategy 2: 57 weeks training and 8 weeks predictions

# COMMAND ----------

preds_train_57wks_test_8wks = mmm.predict(media=media_data_test,
                              extra_features=extra_features_test,
                              seed=SEED)

# COMMAND ----------

plot.plot_out_of_sample_model_fit(out_of_sample_predictions=preds_train_57wks_test_8wks,
                                 out_of_sample_target=target_test)

# COMMAND ----------

# MAGIC %md
# MAGIC MAPE : 9.3

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Testing Strategy 3: 61 weeks training and 4 weeks predictions

# COMMAND ----------

mmm.fit(
    media=media_data_train_61wks,
    media_prior=costs,
    target=target_train_61wks,
    extra_features=extra_features_train_61wks,
    number_warmup=1200,
    number_samples=1000,
    seed=SEED)

# COMMAND ----------

preds_train_61wks_test_4wks = mmm.predict(media=media_data_test_61wks,
                              extra_features=extra_features_test_61wks,
                              seed=SEED)

# COMMAND ----------

plot.plot_out_of_sample_model_fit(out_of_sample_predictions=preds_train_61wks_test_4wks,
                                 out_of_sample_target=target_test_61wks)

# COMMAND ----------

# MAGIC %md
# MAGIC MAPE : 7.5

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Error metric Insights
# MAGIC 
# MAGIC * For all the testing strategies, MAPE is < 10%
# MAGIC * From the above testing it is seen that MAPE is increasing when used for predictions for longer period

# COMMAND ----------

# MAGIC %md
# MAGIC * End of this Notebook
