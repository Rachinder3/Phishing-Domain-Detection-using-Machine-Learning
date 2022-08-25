import sys
from tkinter import E, PROJECTING
from phishing_domain_detection.component.model_trainer import ModelTrainer
from phishing_domain_detection.config.configuration import Configuration
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception

from phishing_domain_detection.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact


from phishing_domain_detection.component.data_ingestion import DataIngestion
from phishing_domain_detection.component.data_validation import DataValidation
from phishing_domain_detection.component.data_transformation import DataTransformation
from phishing_domain_detection.component.model_trainer import ModelTrainer
from phishing_domain_detection.component.model_evaluation import ModelEvaluation
from phishing_domain_detection.component.model_pusher import ModelPusher
import os,sys


class Pipeline:
    def __init__(self, config = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(self.config.get_data_validation_config(),data_ingestion_artifact, self.config.get_training_pipeline_config())

            return data_validation.initiate_data_validation()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            
            return data_transformation.initialize_data_transformation()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact, model_trainer_config=self.config.get_model_trainer_config())
            
            return model_trainer.initialize_model_trainer()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact : ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            model_pusher = ModelPusher(model_pusher_config=self.config.get_model_pusher_config(),
                        model_evaluation_artifact=model_evaluation_artifact)
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def run_pipeline(self):
        try:
            ## Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact= data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
            