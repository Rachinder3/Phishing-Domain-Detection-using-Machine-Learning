from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception

from phishing_domain_detection.entity.artifact_entity import  ModelEvaluationArtifact, ModelPusherArtifact

from phishing_domain_detection.entity.config_entity import ModelPusherConfig

import os,sys
import shutil


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact) -> None:
        try:
            logging.info(f"{'>>'*30} Model Push Started {'>>'*30}")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def export_model(self):
        try:
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            
            export_dir = self.model_pusher_config.export_dir_path
            
            model_file_name = os.path.basename(evaluated_model_file_path)
            
            exported_model_file_path = os.path.join(export_dir, model_file_name)
            
            logging.info(f"Exporting model to [ {exported_model_file_path} ]")
            
            os.makedirs(export_dir,exist_ok=True)
            
            shutil.copy(src=evaluated_model_file_path, dst=exported_model_file_path) 
            # Can push the model object into some cloud. boto3 library to be used. Can create 2 functions in utils,
            # push into s3 bucket and fetch from s3 bucket.
                       
            logging.info(f"Trained model kept in : {evaluated_model_file_path} is copied to export dir: [ {exported_model_file_path} ] " )
            
            model_pusher_artifact = ModelPusherArtifact(
                is_model_pushed=True,
                export_model_file_path=exported_model_file_path
                
            )
            
            logging.info(f"Model pusher artifact : [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise Phishing_Exception(e,sys) from e    
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'>>'*30} Model Push Completed {'>>'*30}")
    
