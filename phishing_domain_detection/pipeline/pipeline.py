from phishing_domain_detection.component.model_trainer import ModelTrainer
from phishing_domain_detection.config.configuration import Configuration
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception

from phishing_domain_detection.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact


from phishing_domain_detection.component.data_ingestion import DataIngestion
from phishing_domain_detection.component.data_validation import DataValidation
from phishing_domain_detection.component.data_transformation import DataTransformation
from phishing_domain_detection.component.model_trainer import ModelTrainer
from phishing_domain_detection.component.model_evaluation import ModelEvaluation
from phishing_domain_detection.component.model_pusher import ModelPusher
import os,sys


from collections import namedtuple
from datetime import datetime
import uuid
from phishing_domain_detection.constants import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME, CURRENT_TIME_STAMP


from threading import Thread
from typing import List

from multiprocessing import Process

import pandas as pd

Experiment = namedtuple("Experiment",[
    "experiment_id","initialization_timestamp", "artifact_timestamp",
    "running_status", "start_time","stop_time","execution_time", "message", 
    "experiment_file_path", "custom_threshold","model_f1","model_recall",
    "model_precision","is_model_accepted"
])

class Pipeline(Thread):
    ### declaring class level experiment attributes
    experiment = Experiment(*([None]*14)) # declaring all attributes of this class level object as None
    experiment_file_path = None # declare the experiment file path as None as of Now
    
    def __init__(self, config = Configuration()) -> None:
        try:
            
            self.config = config
            
            os.makedirs(self.config.training_pipeline_config.artifact_dir, exist_ok=True)
            ## getting path of experiment file
            Pipeline.experiment_file_path = os.path.join(self.config.training_pipeline_config.artifact_dir, EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            
            super().__init__(daemon=False, name='Pipeline') # calling constructor of parent class
            
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
            
            ## If pipeline is already running, then don't start another pipeline
            if Pipeline.experiment.running_status: ## Pipeline already running, so don't run another pipeline
                logging.info(f"Pipeline already running, skipping this pipeline run.")
                return Pipeline.experiment

            logging.info(f"Pipeline starting")
            
            ## generating unique id for this pipeline run
            experiment_id = str(uuid.uuid4())
            
            ## Updating the experiment (which is a class) attribute
            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                artifact_timestamp=CURRENT_TIME_STAMP,
                initialization_timestamp=CURRENT_TIME_STAMP,
                running_status=True,
                start_time=datetime.now(),
                stop_time=None,
                custom_threshold=None,
                experiment_file_path=Pipeline.experiment_file_path,
                execution_time=None,
                message=f"Pipeline has been started successfully",
                is_model_accepted=None,
                model_f1=None,
                model_precision=None,
                model_recall=None
            )
            
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            
            self.save_experiment()
            
            
            
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
            
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Model Pusher artifact: {model_pusher_artifact}")
            
            else:
                logging.info("Trained model rejected")

            logging.info(f"Pipeline Run Completed")

            stop_time = datetime.now()
            
            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                initialization_timestamp= CURRENT_TIME_STAMP,
                artifact_timestamp=CURRENT_TIME_STAMP,
                running_status=False,
                start_time=Pipeline.experiment.start_time,
                stop_time=stop_time,
                custom_threshold = model_trainer_artifact.custom_threshold,
                execution_time= stop_time - Pipeline.experiment.start_time,
                experiment_file_path= Pipeline.experiment.experiment_file_path,
                is_model_accepted=model_evaluation_artifact.is_model_accepted,
                message=f"Pipeline has finished executing.",
                model_f1=model_trainer_artifact.model_f1,
                model_precision=model_trainer_artifact.model_precision,
                model_recall=model_trainer_artifact.model_recall
                
            )
            
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    
    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is None:
                print("Trigger the pipeline atleast once")
                logging.info("Trigger the pipeline first")
                
            experiment = Pipeline.experiment
            
            experiment_dict = experiment._asdict()
            
            experiment_dict = {key:[value] for key,value in experiment_dict.items()}
            # keys serve as column names, value is enclosed within [], to create a list of values for easy creation of dataframe
            
            experiment_dict.update(
                {
                    "created_time_stamp":[datetime.now()],
                    "experiment_file_path": [os.path.basename(Pipeline.experiment.experiment_file_path)]
                }
            )
            
            experiment_report = pd.DataFrame(experiment_dict)
            
            os.makedirs(os.path.dirname(Pipeline.experiment.experiment_file_path), exist_ok=True)
            
            if os.path.exists(Pipeline.experiment.experiment_file_path):
                ## Not the first experiment
                experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
            else:
                ## 1st experiment
                experiment_report.to_csv(Pipeline.experiment_file_path, mode="w", index=False, header=True) 
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
        
    @classmethod
    def get_experiments_status(cls, rows_to_return = 5) ->pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                return df.iloc[-1:(-1*rows_to_return)-1:-1].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                return pd.DataFrame()  # empty dataframess
        except Exception as e:
            raise Phishing_Exception(e,sys) from e