import sys
from tkinter import E
from phishing_domain_detection.config.configuration import Configuration
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception

from phishing_domain_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from phishing_domain_detection.entity.config_entity import DataIngestionConfig

from phishing_domain_detection.component.data_ingestion import DataIngestion
from phishing_domain_detection.component.data_validation import DataValidation
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
    
    def start_data_transformation(self):
        pass
    
    def start_model_trainer(self):
        pass
    
    def stat_model_evaluation(self):
        pass
    
    def run_pipeline(self):
        try:
            ## Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
            