## Reads the config yaml to know various configurations, populates name tuple entities for individual components so that they can be fed to the components


import imp
from tkinter import E
from phishing_domain_detection.entity.config_entity import *
from phishing_domain_detection.util.util import read_yaml_file
import os,sys
from phishing_domain_detection.constants import *
from phishing_domain_detection.exception import Phishing_Exception
from phishing_domain_detection.logger import logging


class Configuration:
    
    def __init__(self,
                config_file_path:str = CONFIG_FILE_PATH,
                current_time_stamp : str =  CURRENT_TIME_STAMP,
                ) -> None:
        """ Initializer function for configuration class

        Args:
            config_file_path (str, optional): path to the config.yaml file. Defaults to CONFIG_FILE_PATH.
            current_time_stamp (str, optional): Current Timestamp. Defaults to CURRENT_TIME_STAMP.

        Raises:
            Phishing_Exception: Our custom exception
        """
        
        try:
            
            self.config_info = read_yaml_file(config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = CURRENT_TIME_STAMP
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir  ## getting the artifact directory
            
            ## Artifact directory for Data Ingestion
            data_ingestion_artifact_dir = os.path.join(
                artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR_KEY,
                CURRENT_TIME_STAMP
            )
            print(data_ingestion_artifact_dir)
            # Getting config for data ingestion 
            data_ingestion_config = self.config_info[DATA_INGESTION_CONFIG_KEY]
            
            # Getting dataset directory
            dataset_dir = data_ingestion_config[DATA_INGESTION_DATASET_DIR_KEY]
            

            # Getting dataset name
            dataset_name  = data_ingestion_config[DATA_INGESTION_DATASET_NAME_KEY]
            
            # Getting table name
            table_name = data_ingestion_config[DATA_INGESTION_TABLE_NAME_KEY]
            
            ''' # Building path for dataset
            dataset_path = os.path.join(
                dataset_dir,
                dataset_name
            )'''

            ## Getting the top features
            top_features = data_ingestion_config[DATA_INGESTION_TOP_FEATURES_KEY]
            

            ingested_data_dir = os.path.join(data_ingestion_artifact_dir, 
                                              data_ingestion_config[DATA_INGESTION_INGESTED_DIR_KEY])
            

            # Path where training data will be stored
            ingested_train_dir = os.path.join(ingested_data_dir,
                                               data_ingestion_config[DATA_INGESTION_INGESTED_TRAIN_DIR_KEY])
            
            
            # Path where testing data will be stored
            ingested_test_dir = os.path.join(ingested_data_dir,
                                             data_ingestion_config[DATA_INGESTION_INGESTED_TEST_DIR_KEY])
            
            # Populating config_entity with our custom config
            data_ingestion_config = DataIngestionConfig(
                dataset_dir=dataset_dir,
                dataset_name = dataset_name,
                table_name=table_name,
                top_features=top_features,
                ingested_train_dir=ingested_train_dir,
                ingested_test_dir=ingested_test_dir
            )
            
            logging.info(f"Data Ingestion Config: {data_ingestion_config}")
            return data_ingestion_config
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
        
    
    def get_data_validation_config(self) -> DataValidationConfig:
        pass
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        pass
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        pass
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        pass
    
    def get_model_pusher_config(self) -> ModelExportConfig:
        pass
    
    
    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """ Get Training pipeline config entity and populate with our custom congig

        Raises:
            Phishing_Exception: our custom exception

        Returns:
            TrainingPipelineConfig: config entity populated with our custom config
        """
        
        try:
            ## Get training pipeline config
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            
            ## Building artifact dir
            artifact_dir = os.path.join(
                ROOT_DIR,
                training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            )
            
            ## Populating the entity with our values
            training_pipeline_config =  TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    