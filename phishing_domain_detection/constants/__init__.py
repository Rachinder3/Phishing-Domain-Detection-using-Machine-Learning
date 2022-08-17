## storing all the constants at one place

import os
from datetime import datetime


ROOT_DIR = os.getcwd() # Working directory of app.py

CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)


CURRENT_TIME_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

## Training Pipeline related variables

TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"


## Data Ingestion related variables
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config" # Upper level for getting dictionary for data ingestion
DATA_INGESTION_ARTIFACT_DIR_KEY = "data_ingestion" # Special variable, Directory where we store the artifacts with respect to Data Ingestion
DATA_INGESTION_DATASET_DIR_KEY  = "dataset_dir"
DATA_INGESTION_DATASET_NAME_KEY = "dataset_name"
DATA_INGESTION_TABLE_NAME_KEY = "table_name"
DATA_INGESTION_TOP_FEATURES_KEY = "top_features"
DATA_INGESTION_INGESTED_DIR_KEY = "ingested_dir"
DATA_INGESTION_INGESTED_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_INGESTED_TEST_DIR_KEY = "ingested_test_dir"



## Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_ARTIFACT_DIR_NAME = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"



