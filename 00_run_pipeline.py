# Databricks notebook source
# MAGIC %md 
# MAGIC ## Objective:
# MAGIC * This notebook helps in running sequentially all the notebooks required for offline attribution model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC ### 1. Install Dependencies
# MAGIC ### 2. Run the notebooks

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Install Dependencies <a id='InstallDependencies'></a>

# COMMAND ----------

dbutils.library.installPyPI('mlflow', version='2.2.2')
dbutils.library.installPyPI('openpyxl', version='3.1.2')
dbutils.library.installPyPI('plotly', version='5.1.0')
dbutils.library.installPyPI('hyperopt', version='0.2.7')
!pip install lightweight-mmm==0.1.5
dbutils.library.restartPython()

# COMMAND ----------

# Configure widgets

dbutils.widgets.removeAll()
dbutils.widgets.text("file_location", "/dbfs/FileStore/FileStore/offline_attribution/datasets/")
file_location = dbutils.widgets.getArgument(name='file_location')
dbutils.widgets.text("visualisation_plot_path", "/dbfs/FileStore/FileStore/offline_attribution/datasets/visualisation_plot/")
visualisation_plot_path = dbutils.widgets.getArgument(name='visualisation_plot_path')
dbutils.widgets.text("experiment_path", "/Shared/Analytics/offline_attribution/offline_attribution_experiments")
experiment_path = dbutils.widgets.getArgument(name='experiment_path')
dbutils.widgets.text("train_test_data_file", "/dbfs/FileStore/FileStore/offline_attribution/datasets/offline_attribution_train_test_data.csv")
train_test_data_file = dbutils.widgets.getArgument(name='train_test_data_file')

# COMMAND ----------

# Import all the necessary libraries

import numpy as np
import pandas as pd
import mlflow
from  mlflow.tracking import MlflowClient

# COMMAND ----------

# # Create new experiment (Only for the first time)
# experiment_id = mlflow.create_experiment("/Shared/Analytics/offline_attribution/offline_attribution_experiments")
# experiment = mlflow.get_experiment(experiment_id)

client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_path)
ex_id = experiment.experiment_id

# COMMAND ----------

# Display settings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 100)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2. Run the Notebooks <a id='RuntheNotebooks'></a>

# COMMAND ----------

# MAGIC %run ./01_dataset_preparation

# COMMAND ----------

# MAGIC %run ./02_eda

# COMMAND ----------

# MAGIC %run ./03_initial_model_train_test

# COMMAND ----------

# MAGIC %run ./04_final_model_with_recommendation

# COMMAND ----------

# MAGIC %md
# MAGIC * End of Notebook
