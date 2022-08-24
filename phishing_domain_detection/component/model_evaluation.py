from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception

from phishing_domain_detection.entity.config_entity import ModelEvaluationConfig
from phishing_domain_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact

from phishing_domain_detection.constants import *

import numpy as np
import os,sys
import pandas as pd

from phishing_domain_detection.entity.model_factory import evaluate_classification_report

from phishing_domain_detection.util.util import *


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact) -> None:
        try:
            logging.info(f"{'>>'*30} Model Evaluation Started. {'>>'*30}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    
    def get_best_model(self):
        """Gets the model in production

        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            None if no model in production else the model object currently in production
        """
        try:
            model = None
            
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            
            ## checking if file exists, if not, it means there are no model in production so create blank file
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path) ## creating the model evaluation artifact directory
                
                return model
            
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
            
            if BEST_MODEL_KEY not in model_eval_file_content:
                return model ## Blank file, no best model available
            
            model = load_object(file_path= model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
            
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e    
    
    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            ### model eval path
            model_eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            ### loading the model eval file
            model_eval_content = read_yaml_file(file_path=model_eval_file_path)
            
            model_eval_content  = dict() if model_eval_content is None else model_eval_content
            
            previous_best_model = None
            
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]
            
            logging.info(f"Previous eval results : {previous_best_model}")
            
            eval_result = {
                BEST_MODEL_KEY:{
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path
                }
            }
            
            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
            
            model_eval_content.update(eval_result)
            logging.info(f"Updated model evaluation file : {model_eval_content}")
            
            write_yaml_file(file_path=model_eval_file_path, data= model_eval_content)    
            
            
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            ## Getting path of model object
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            ## Loading object of model
            trained_model_object = load_object(file_path= trained_model_file_path)
            
            ## Getting train and test file paths
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
        
            ## Getting schema file path to know target column
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(file_path=schema_file_path)
            
            ### Loading the train and test file
            train_dataframe:pd.DataFrame = load_data(file_path=train_file_path, schema_file_path=self.data_validation_artifact.schema_file_path)
            test_dataframe:pd.DataFrame = load_data(file_path=test_file_path, schema_file_path=self.data_validation_artifact.schema_file_path)
            
            ## Getting the target column
            target_column_name  = schema[SCHEMA_TARGET_COLUMN_KEY]
            
            logging.info(f"Converting dataframes to numpy arrays")
            
            ### Extracting target and converting to numpy array
            
            train_target_array = np.array(train_dataframe[target_column_name])
            test_target_array = np.array(test_dataframe[target_column_name])
            
            logging.info(f"Conversion complete of target column to numpy array")
            
            ### Dropping target columns
            
            train_dataframe.drop(target_column_name, axis = 1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            
            logging.info("Dropped target column from training and testing data")
            
            
            model = self.get_best_model()
            
            if model is None:
                logging.info(f"Not found any existing model. Hence accepting the trained model.")
                
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )
            
                self.update_evaluation_report(model_evaluation_artifact= model_evaluation_artifact)
                logging.info(f"Model accepted. Model evaluationa artifact : {model_evaluation_artifact}")
                
                return model_evaluation_artifact
            
            
            model_list = [model, trained_model_object]
            
            metric_info_artifact = evaluate_classification_report(
                model_list= model_list,
                custom_threshold= self.model_trainer_artifact.custom_threshold,
                X_train= train_dataframe,
                y_train= train_target_array,
                X_test=test_dataframe,
                y_test=test_target_array,
                base_precision = self.model_trainer_artifact.base_precision,
                base_recall=self.model_trainer_artifact.model_recall,
            )
            
            logging.info(f"Model Evaluation Complete. Model metric artifact : {metric_info_artifact}")
            
            if metric_info_artifact is None:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path
                )
                logging.info(model_evaluation_artifact)
                return model_evaluation_artifact
            
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact= model_evaluation_artifact)
                logging.info(f"Model accepted. Model Eval artifact : {model_evaluation_artifact}")
            else:
                logging.info(f"Trained model is no better than existing model hence not accepting trained model")    
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path
                )
                
            return model_evaluation_artifact           
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*30} Model Evaluation Finished. {'>>'*30}")
        
        