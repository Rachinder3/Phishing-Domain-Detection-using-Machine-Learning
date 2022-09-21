
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception

from phishing_domain_detection.entity.model_factory import *
from phishing_domain_detection.entity.config_entity import ModelTrainerConfig

import os, sys

from phishing_domain_detection.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact


from phishing_domain_detection.util.util import load_numpy_array_data, save_object, load_object



class PhishingEstimator:
    def __init__(self, preprocessed_object, trained_model_object, custom_threshold) -> None:
        try:
            self.preprocessed_object = preprocessed_object
            self.trained_model_object = trained_model_object
            self.custom_threshold = custom_threshold
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def predict(self, X):
        """Does predictions on the basis of custom threshold

        Args:
            X (_type_): Input Dataframe

        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            _type_: Predictions
        """
        
        try:
            transformed_features = self.preprocessed_object.transform(X)
            
            prediction_probabilities = self.predict_proba(X)[:,1]
            

            predictions = np.where(prediction_probabilities>self.custom_threshold,1,0)
            
            
            
            return predictions
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
        
    def predict_proba(self,X):
        """Returns probabilities instead of predictions

        Args:
            X (_type_): Input features

        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            _type_: Predicted probabilities
        """
        try:
            transformed_features = self.preprocessed_object.transform(X)
            
            prediction_probabilities = self.trained_model_object.predict_proba(transformed_features)
            
            return prediction_probabilities
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    



class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact,
                model_trainer_config : ModelTrainerConfig) -> None:
        try:
            logging.info(f"{'>>'*30} Model Trainer Log started {'>>'*30}")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e

    def initialize_model_trainer(self) -> ModelTrainerArtifact:
        try:
            ## Loading data
            
            logging.info("Loading Training data")
            transformed_training_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(transformed_training_file_path)

            logging.info("Loading Testing data")
            transformed_testing_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(transformed_testing_file_path)

            logging.info("Splitting Independant and Dependant Features")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            logging.info("Extract path of model config")
            model_config_file_path = self.model_trainer_config.model_config_file_path
            
            logging.info(f"Initializing Model Factory class with file :{model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_recall = self.model_trainer_config.base_recall
            base_precision  = self.model_trainer_config.base_precision
            
            logging.info(f"Expected Recall : {base_recall} and Expected Precision : {base_precision}")
            
            logging.info(f"Initiating operation model selection")

            best_model = model_factory.get_best_model(X = X_train, y=y_train, scoring_parameter_for_grid_search_cv=self.model_trainer_config.scoring_parameter_for_grid_search_cv, base_recall= self.model_trainer_config.base_recall)
            
            logging.info(f"Best model on training dataset : {best_model}")
            
            logging.info("Extracting trained model list.")
            
            gird_searched_best_model_list:List[GridSearchBestModel] = model_factory.grid_searched_best_model_list
            
            model_list = [model.best_model for model in gird_searched_best_model_list]  
            logging.info(f"Evaluating all the models on training and testing data both")
            
            metric_info: MetricInfoArtifact = evaluate_classification_report(model_list= model_list, custom_threshold= self.model_trainer_config.custom_threshold, X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test, base_precision=self.model_trainer_config.base_precision, base_recall=self.model_trainer_config.base_recall)
            
            preprocessing_obj = load_object(file_path= self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object
            
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            
            phishing_model = PhishingEstimator(preprocessed_object=preprocessing_obj, trained_model_object=model_object, custom_threshold=self.model_trainer_config.custom_threshold)
            
            logging.info(f"Saving the model object at: {trained_model_file_path}")
            save_object(trained_model_file_path,phishing_model)
            
            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model Trained Successfully",
                trained_model_file_path=trained_model_file_path,
                train_recall=metric_info.train_recall,
                test_recall=metric_info.test_recall,
                train_precision=metric_info.train_precision,
                test_precision=metric_info.test_precision,
                model_f1=metric_info.model_f1,
                custom_threshold=self.model_trainer_config.custom_threshold,
                model_recall= metric_info.model_recall,
                model_precision= metric_info.model_precision,
                base_recall= base_recall,
                base_precision= base_precision
            )
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        except Exception as e:
            raise Phishing_Exception(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30} Model Trainer Log Completed {'>>'*30}")