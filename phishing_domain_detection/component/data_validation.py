from cmath import e
import imp
from re import I
from tkinter import E
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception
from phishing_domain_detection.config.configuration import Configuration
from phishing_domain_detection.entity.config_entity import DataValidationConfig
from phishing_domain_detection.entity.artifact_entity import DataIngestionArtifact
import sys,os
import pandas as pd
from phishing_domain_detection.util.util import *
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json
from phishing_domain_detection.entity.config_entity import TrainingPipelineConfig
from phishing_domain_detection.constants import *
import datetime
import re
from phishing_domain_detection.entity.artifact_entity import DataValidationArtifact

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                data_ingestion_artifact: DataIngestionArtifact,
                training_pipeline_config : TrainingPipelineConfig) -> None:
        try:
            logging.info(f"{'>>'*20} Data Validation Log Started {'>>'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.training_pipeline_config = training_pipeline_config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    ### Validation functions
    
    def does_train_test_file_exists(self)->None:
        try:
            logging.info("Checking if Training and Testing Files exist")

            ## Getting the training file path
            train_file_path = self.data_ingestion_artifact.train_file_path
            
            ## Getting the testing file path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            
            ## Checking if training file exists
            is_train_file_exist = os.path.exists(train_file_path)
            
            ## Checking if testing file exists
            is_test_file_exist = os.path.exists(test_file_path)
            
            is_available = is_test_file_exist and is_train_file_exist
            
            logging.info(f"Do Training and Testing files exist ?? -> {is_available}")
            if not is_available:
                message = f"Training file : {train_file_path} or Testing File: {test_file_path} doesn't exist"
                logging.info(message)
                raise Exception(message)
                           
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def load_training_and_testing_files(self):
        try:
            
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df, test_df
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    
    def validate_schema(self) -> None:
        try: 
            ## Checks:
            ##  1. Number of columns
            ##  2. Check column names are same
            ##  3. Check domain values of categorical features (we don't have any)

            
            logging.info("Starting schema validation")
            ## Loading the schema
            
            desired_schema = read_yaml_file(self.data_validation_config.schema_file_path)

            ## Getting train and test files
            
            train_df, test_df = self.load_training_and_testing_files()

            ## Check 1. Number of columns:
            training_file_is_num_cols_same =  len(train_df.columns) == len(desired_schema['columns'])+1
            
            testing_file_is_num_cols_same = len(test_df.columns) == len(desired_schema['columns'])+1
            
            is_num_cols_same = training_file_is_num_cols_same and testing_file_is_num_cols_same
            
            message = f"Training File Exists ?? -> {training_file_is_num_cols_same} ,  testing file exists ?? -> {testing_file_is_num_cols_same}"
            logging.info(message)
            
            
            ## Check 2. Column names:
            
            desired_schema_with_target = set(desired_schema['columns'])
            desired_schema_with_target.add(desired_schema['target_column'])
            
            
            training_file_is_column_names_same = set(train_df.columns) == desired_schema_with_target
            testing_file_is_column_names_same = set(test_df.columns) == desired_schema_with_target
            
            message = f"Does Training file have desired schema ?? -> {training_file_is_column_names_same} , Does Testing File have desired schema ? -> {testing_file_is_column_names_same}"
            
            column_names_same = training_file_is_column_names_same and testing_file_is_column_names_same
            
            logging.info(message)
            
            ## Final Check
            
            final_checks_passed = is_num_cols_same and column_names_same
            
            if not final_checks_passed:
                message = "Schema checks not passed"
                logging.info(message)
                raise Exception(message)
            
            
            logging.info("Final Checks passed")
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def get_oldest_train_file(self):
        try:
            logging.info("Finding oldest file")
            data_ingestion_dir = os.path.join(
                self.training_pipeline_config.artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR_KEY
            )
            
            ingestions = os.listdir(data_ingestion_dir)
            
            ingestions = [datetime.datetime.strptime(ingestion,'%Y-%m-%d_%H-%M-%S') for ingestion in ingestions]
            
            oldest_time_stamp = min(ingestions)
            oldest_time_stamp = datetime.datetime.strftime(oldest_time_stamp,'%Y-%m-%d_%H-%M-%S')
            
            
            oldest_train_file_path = re.sub(pattern="\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",string=self.data_ingestion_artifact.train_file_path, repl=oldest_time_stamp)
            
            logging.info(f"Oldest file path : {oldest_train_file_path}")
            oldest_train_file = pd.read_csv(oldest_train_file_path)
            
            return oldest_train_file
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def get_and_save_data_drift_report(self):
        try:
            logging.info("Gnerating Drift Report")
            profile = Profile(sections=[DataDriftProfileSection()])
            
            train_df, _ = self.load_training_and_testing_files()
            
            oldest_df = self.get_oldest_train_file()
            
            profile.calculate(oldest_df, train_df)
            
            report = json.loads(profile.json())
            
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            
            logging.info(f"Drift report saved at : {report_file_path}") 
            
            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)
            return report
        
                        
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def get_and_save_data_drift_report_page(self):
        try:
            logging.info(f"Generating Drift Report HTML page")
            
            
            dashboard = Dashboard(tabs = [DataDriftTab()])
            
            train_df,_ = self.load_training_and_testing_files()
            
            oldest_df = self.get_oldest_train_file()
            
            dashboard.calculate(oldest_df, train_df)
            
            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)
            
            dashboard.save(report_page_file_path)
            
            logging.info(f"Drift Report Page saved at: {report_page_file_path}")
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
        
    
    
    def is_data_drift_found(self):
        try:
            report = self.get_and_save_data_drift_report()
            
            is_data_drift_detected = report['data_drift']['data']['metrics']['dataset_drift']
            
            
            
            if is_data_drift_detected:
                message = f"Data Drift has been detected. Shutting down the pipeline."
                logging.info(message)
                raise Exception(message)
            
            self.get_and_save_data_drift_report_page()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e 
    
    
    #### Actual Validation
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.does_train_test_file_exists()
            self.validate_schema()
            self.is_data_drift_found()
            
            data_validation_artifact = DataValidationArtifact(
                schema_file_path= self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successfully"
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            
        except Exception as e:
            raise Phishing_Exception(e, sys) from e

    
    def __del__(self):
        logging.info(f"{'>>'*20} Data Validation Log Completed {'>>'*20}")