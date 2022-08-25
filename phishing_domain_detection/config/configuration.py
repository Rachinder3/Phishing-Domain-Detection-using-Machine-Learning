## Reads the config yaml to know various configurations, populates name tuple entities for individual components so that they can be fed to the components


import imp
from tkinter import E
from phishing_domain_detection.entity.config_entity import *
from phishing_domain_detection.util.util import read_yaml_file
import os,sys
from phishing_domain_detection.constants import *
from phishing_domain_detection.exception import Phishing_Exception
from phishing_domain_detection.logger import logging
from datetime import datetime

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
       
        try:

            
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            
            ## Path for data validation artifact dir
            data_validation_artifact_dir = os.path.join(
                artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR_NAME,
                CURRENT_TIME_STAMP
            )
            
            ## Building the schema file path
            
            schema_file_path = os.path.join(ROOT_DIR,
                         data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                         data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
            
            
            ## Building the report paths
            report_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            )
            
            report_page_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            )
            
            
            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                report_file_path = report_file_path,
                report_page_file_path= report_page_file_path
            )
            return data_validation_config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e            
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            
            data_transformation_config = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            data_transformation_artifact_dir = os.path.join(
                artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR_NAME,
                CURRENT_TIME_STAMP
            )
            
            use_box_transformation = data_transformation_config[DATA_TRANSFORMATION_USE_BOX_COX_KEY]
            
            transformed_train_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],
                data_transformation_config[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY]
            )
            
            transformed_test_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],
                data_transformation_config[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY]
            )
            
            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME_KEY]
            )
            
            data_transformation_config = DataTransformationConfig(
                use_box_cox_transformation=use_box_transformation,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir,
                preprocessed_object_file_path=preprocessed_object_file_path
                
            )
            
            logging.info(f"Data Transformation config : {data_transformation_config}")
            
            return data_transformation_config
        
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            
            model_trainer_config = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            model_trainer_artifact_dir = os.path.join(
                artifact_dir,
                MODEL_TRAINER_ARTIFACT_DIR,
                CURRENT_TIME_STAMP
            )
            
            trained_model_file_path = os.path.join(
                model_trainer_artifact_dir,
                model_trainer_config[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],
                model_trainer_config[MODEL_TRAINER_MODEL_FILE_NAME_KEY]
            )
            
            scoring_parameter_for_grid_search_cv = model_trainer_config[MODEL_TRAINER_SCORING_PARAMETER_FOR_GRID_SEARCH_CV_KEY]
            
            base_precision = model_trainer_config[MODEL_TRAINER_BASE_PRECISION_KEY]
            
            base_recall = model_trainer_config[MODEL_TRAINER_BASE_RECALL_KEY]
            
            model_config_file_path = os.path.join(
                model_trainer_config[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                model_trainer_config[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]
            )
            
            custom_threshold = model_trainer_config[MODEL_TRAINER_CUSTOM_THRESHOLD]
            
            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                scoring_parameter_for_grid_search_cv=scoring_parameter_for_grid_search_cv,
                base_precision=base_precision,
                base_recall=base_recall,
                model_config_file_path=model_config_file_path,
                custom_threshold= custom_threshold
            )
            
            logging.info(f"Model trainer config: {model_trainer_config}")
            
            return model_trainer_config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            model_evalutaion_artifact_dir  = os.path.join(
                artifact_dir,
                
                MODEL_EVALUATION_CONFIG_ARTIFACT_DIR_KEY
            )
            
            model_evaluation_file_path = os.path.join(model_evalutaion_artifact_dir, model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY])
            
            model_evaluation_config = ModelEvaluationConfig(
                model_evaluation_file_path=model_evaluation_file_path,
                time_stamp=CURRENT_TIME_STAMP
            )
            
            logging.info(f"Model Evaluation Config : {model_evaluation_config}")
            return model_evaluation_config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def get_model_pusher_config(self) -> ModelPusherConfig:
        try:
            time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
            model_pusher_Config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            
            export_dir_path = os.path.join(ROOT_DIR,
                                           model_pusher_Config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           time_stamp)  ## Will create this in the root directory
            
            model_pusher_config = ModelPusherConfig(
                export_dir_path=export_dir_path
            )
            
            logging.info(f"Model Pusher Config: {model_pusher_config}")
            
            return model_pusher_config
        
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    
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
    