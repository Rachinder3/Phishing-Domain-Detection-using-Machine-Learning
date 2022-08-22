import imp
from pyexpat import model
import importlib
from cmath import e, log
from typing import List

import numpy as np

import yaml

from phishing_domain_detection.exception import Phishing_Exception

import os,sys

from collections import namedtuple

from phishing_domain_detection.logger import logging

from sklearn.metrics import f1_score, recall_score, precision_score

GRID_SEARCH_KEY = 'grid_search'

MODULE_KEY = 'module'

CLASS_KEY = "class"

PARAM_KEY = "params"

MODEL_SELECTION_KEY = "model_selection"

SEARCH_PARAM_GRID_KEY = "search_param_grid"



InitializedModelDetail = namedtuple("InitializedModelDetail",[
    "model_serial_number","model","model_name","param_grid_search"
])


GridSearchBestModel = namedtuple("GridSearchBestModel",["model_serial_number",
                                                        "model",
                                                        "best_model",
                                                        "best_paramters",
                                                        "best_score"])

BestModel = namedtuple("BestModel",["model_serial_number",
                                                        "model",
                                                        "best_model",
                                                        "best_paramters",
                                                        "best_score"])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",["model_name", "model_object","train_recall", "test_recall","train_precision","test_precision", "model_f1","index_number"])

def evaluate_classification_report(model_list, custom_threshold,X_train: np.array, y_train:np.array, X_test: np.array, y_test: np.array, base_precision= 0.8, base_recall: float = 0.9):
    try:
        index_number = 0
        metric_info_artifact = None
        
        for model in model_list:
            model_name = str(model) ## Getting name of model from model object
            logging.info(f"{'>>'*30} Starting evaludation model: [ {type(model).__name__} ] {'>>'*30}")
            
            ## Getting predictions on training and testing dataset 
            ##### Training data
            y_train_pred_probabilities = model.predict_proba(X_train)[:,1]
            y_train_pred =  np.where(y_train_pred_probabilities > custom_threshold, 1, 0)
                    
            
            ##### Testing Data
            y_test_pred_probabilities = model.predict_proba(X_test)[:,1]
            y_test_pred = np.where(y_test_pred_probabilities > custom_threshold,1,0)
            
            
            ##### Calculating precision on training data and testing data  
            train_precision = precision_score(y_true = y_train, y_pred = y_train_pred)
            test_precision = precision_score(y_true=y_test, y_pred=y_test_pred)
            avg_precision = (2 * (train_precision*test_precision)) / (train_precision * test_precision)
            
            ##### Calculating recall on training data and testing data
            train_recall = recall_score(y_true = y_train, y_pred = y_train_pred)
            test_recall = recall_score(y_true=y_test, y_pred=y_test_pred)
            avg_recall = (2 * (train_recall*test_recall)) / (train_recall * test_recall)
            
            ##### Calculating f1 score on training data and testing data
            train_f1 = f1_score(y_true = y_train, y_pred = y_train_pred)
            test_f1 =  f1_score(y_true=y_test, y_pred=y_test_pred)
            avg_f1 = (2 * (train_f1*test_f1)) / (train_f1 * test_f1)
            
            ##### difference of f1 scores to check overfitting
            
            diff_test_train_f1 = abs(test_f1 - train_f1)
            
            #### logging all important metrics
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Criteria\t\tTrain Score\t\t Test Score\t\t Avergae Score\t\t")
            logging.info(f"Recall\t\t{train_recall}\t\t {test_recall}\t\t{avg_recall}")
            logging.info(f"Precision\t\t{train_precision}\t\t {test_precision}\t\t{avg_precision}")
            logging.info(f"F1 score\t\t{train_f1}\t\t {test_f1}\t\t{avg_f1}")
            
            
            
            
            ## We accept a model if it has some threshold recall and  precion.
            ## Also, we check that difference b/w testing and training f1 score is less than a threshold so that we ensure, we are not overfitting
            
            
            # print(train_recall,train_precision,test_recall,test_precision)
            if avg_recall >= base_recall and avg_precision >= base_precision and diff_test_train_f1 < 0.1:
                base_recall = avg_recall ## More interested in increasing recall, precision needs to be atleast equal to base precision
                metric_info_artifact = MetricInfoArtifact(
                    model_name=model_name,
                    model_object=model,
                    train_recall=train_recall,
                    test_recall=test_recall,
                    train_precision=train_precision,
                    test_precision = test_precision,
                    model_f1= avg_f1,
                    index_number= index_number
                )
                
                # print(metric_info_artifact)
                logging.info(f"Acceptable model found: {metric_info_artifact}")
                
                
            index_number += 1
        
        if   metric_info_artifact is None:
            logging.info(f"No model with higher recall than  : {base_recall} or no model with higher precision : {base_precision} or the model overfits because difference b/w train and test f1 scores is greater than {0.5}")  
        
        return metric_info_artifact    
    except Exception as e:
        raise Phishing_Exception(e,sys) from e


def get_sample_model_config_yaml_file(export_dir: str):
    """Generates a sample model file. This will help us to do Hyperparamater tuning and get the best model.

    Args:
        export_dir (str): Path where sample model.yaml will be stored

    Raises:
        Phishing_Exception: Custom Exception
    """
    try:
        model_config = {
            GRID_SEARCH_KEY:{
                CLASS_KEY:"GridSearchCV",
                MODULE_KEY:"sklearn.model_selection",
                PARAM_KEY: {
                    "cv":3,
                    "verbose":10
                }
            },
            MODEL_SELECTION_KEY:{
                "module_0":{
                    CLASS_KEY:"ModelClassName",
                    MODULE_KEY:"module_of_model",
                    PARAM_KEY:{
                        "param_name_1":"value1",
                        "param_name_2":"value2"
                    },
                    SEARCH_PARAM_GRID_KEY:{
                        "param_name":['param_value_1','param_value_2']
                    }
                },
            }
            
        }
        
        os.makedirs(export_dir,exist_ok=True)
        export_file_path = os.path.join(export_dir,"model.yaml")
        
        with open(export_file_path,'w') as file:
            yaml.dump(model_config, file)
            
        return export_file_path
    except Exception as e:
        raise Phishing_Exception(e,sys) from e

class ModelFactory:
    def __init__(self, model_config_path:str = None):
        try:
            self.model_config = ModelFactory.read_params(model_config_path)

            self.grid_search_cv_module: str = self.model_config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name : str = self.model_config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_key: dict = self.model_config[GRID_SEARCH_KEY][PARAM_KEY]

            #print(self.model_config[MODEL_SELECTION_KEY])
            self.models_initialization_config: dict = self.model_config[MODEL_SELECTION_KEY]
        

            self.initialized_model_list = None
            
            self.grid_searched_best_model_list = None
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    
    @staticmethod
    def read_params(config_path: str)->dict:
        try:
            with open(config_path,'r') as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    @staticmethod
    def class_for_name(module_name:str,class_name:str):
        """Loads the module and returns its object

        Args:
            module_name (str): Name of module to be loaded
            class_name (str): Class where module is kept

        Raises:
            Phishing_Exception: Custom Excedption
        """
        
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing: from {module_name} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    @staticmethod
    def update_property_of_class(instance_ref:object, property_data:dict):
        """This method updates the properties of model object and sets them to our desired property

        Args:
            instance_ref (object): Model Object
            property_data (dict): The properties that we want to set for our model object

        Raises:
            Phishing_Exception: Our custom exception
        """
        try:
            if not isinstance(property_data,dict):
                raise Exception("property_data parameter required to be a dictionary")
            
            for key,value in property_data.items():
                logging.info(f"Executing:{str(instance_ref)}.{key} = {value}")
                setattr(instance_ref,key,value)
            
            return instance_ref
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """ Initializes the different models passed in Model.yaml and returns the list 
        of initialized model

        Raises:
            Phishing_Exception: Custom Exception
        """
        try:
            initialized_model_list = []
            
            
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                
                
                # Getting the module of model
                model_obj_ref = ModelFactory.class_for_name(class_name=model_initialization_config[CLASS_KEY], module_name=model_initialization_config[MODULE_KEY])

                
                model = model_obj_ref()  # Creating object of class
                
                
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    
                    model = ModelFactory.update_property_of_class(instance_ref= model, property_data= model_obj_property_data)
                
                ## Initializing empty parameters grid. 
                param_gid_search = {}
                if SEARCH_PARAM_GRID_KEY in model_initialization_config:
                    param_gid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
            
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
            
                model_initialization_config = InitializedModelDetail(
                    model_serial_number=model_serial_number,
                    model=model,
                    model_name=model_name,
                    param_grid_search=param_gid_search,
                )
            
                initialized_model_list.append(model_initialization_config)
            
            self.initialized_model_list = initialized_model_list
        
            return self.initialized_model_list
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
         
        
    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, 
                                          input_feature, output_feature, scoring_paramter:str)-> GridSearchBestModel:
        """This Function executes the grid search cv operation to get best parameter for input and output features based on the scoring parameter

        Args:
            initialized_model (InitializedModelDetail): Initial NModel Object
            input_feature (_type_): Independant features
            output_feature (_type_): Dependant features
            scoring_paramter (str): Scoring to be used in Grid Search CV

        Raises:
            Phishing_Exception: Custom Exeption

        Returns:
            GridSearchBestModel: The best model from Grid Search CV 
        """
        
        try:
                # initial grid search cv object
                
            grid_search_cv_obj_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module, class_name= self.grid_search_class_name)
            grid_search_cv =    grid_search_cv_obj_ref(initialized_model.model, param_grid = initialized_model.param_grid_search, scoring = scoring_paramter)
                
            message = f"{'>>'*30} Grid Search for {type(initialized_model.model).__name__} started. {'>>'*30}"      

            logging.info(message)
            
            grid_search_cv.fit(input_feature,output_feature)
            
            message = f"{'>>'*30} Grid Search for {type(initialized_model.model).__name__} Completed. {'>>'*30}"  
            
            logging.info(message)
            
            grid_search_best_model = GridSearchBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model=initialized_model.model,
                best_model= grid_search_cv.best_estimator_,
                best_paramters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_
            )
            
            return grid_search_best_model
                
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    
    def initialize_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                               input_feature, output_feature, scoring_paramter) -> GridSearchBestModel:
        """This function will perform paramter search operation and
        it will return the best optimistic  model with best paramter

        Args:
            initialized_model (InitializedModelDetail): The initialized model
            input_feature (_type_): The input feature
            output_feature (_type_): The output feature
            scoring_paramter (_type_): Scoring parameter for Grid Search CV

        Raises:
            Phishing_Exception: The custom exception

        Returns:
            GridSearchBestModel: Best paramters for this initialized model
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model, input_feature=input_feature, output_feature= output_feature, scoring_paramter= scoring_paramter)
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    def initialize_best_paramter_search_for_initialized_models(self, initialized_model_list: List[InitializedModelDetail], input_feature, output_feature, scoring_parameter) -> List[GridSearchBestModel]:
        """Executes Grid Search CV on all the models present in Config.yaml / Initialized model list

        Args:
            initialized_model_list (List[InitializedModelDetail]): The Initialized Model List
            input_feature (_type_): Input Features
            output_feature (_type_): Output Features
            scoring_parameter (_type_): Scoring Parameter

        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            _type_: Returns a list of best model after Grid Search CVfor all initialized models
        """
        try:
            self.grid_searched_best_model_list = []
            
            for initialized_model in initialized_model_list:
                grid_search_best_model = self.initialize_best_parameter_search_for_initialized_model(initialized_model= initialized_model, input_feature=input_feature, output_feature=output_feature, scoring_paramter=scoring_parameter)
            
                self.grid_searched_best_model_list.append(grid_search_best_model)
            
            return self.grid_searched_best_model_list
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    @staticmethod   
    def get_model_detail(model_deatils: List[InitializedModelDetail], model_serial_no:str) -> InitializedModelDetail:
        
        try:
            for model_data in model_deatils:
                if model_data.model_serial_number == model_serial_no:
                    return model_data
                
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_search_best_model_list : List[GridSearchBestModel], base_recall = 0.9) -> BestModel:
        """Gets the best model out of all the models present in grid_search_cv list

        Args:
            grid_search_best_model_list (List[GridSearchBestModel]): List of all Grid Searched Models
            base_recall (float, optional): Minimum desired recall. Defaults to 0.9.

        Raises:
            Exception: Custom Exception

        Returns:
            _type_: Best Model out of all the models
        """
        try:
            best_model = None
            for grid_searched_model in grid_search_best_model_list:
                if base_recall < grid_searched_model.best_score:
                    logging.info(f"Acceptable model found: {grid_searched_model}")
                    
                    base_recall = grid_searched_model.best_score
                    
                    best_model = grid_searched_model
            
            if not best_model:
                raise Exception(f"None of the models have base recall: {base_recall}")
            
            logging.info(f"Best Model: {best_model}")
            return best_model
                
                
        except Exception as e:
            pass 
    
    def get_best_model(self, X,y, scoring_parameter_for_grid_search_cv:str,base_recall = 0.9) -> BestModel:
        """Main function of Model Factory script. Simply run this for getting the best model

        Args:
            X (_type_): Independant Features
            y (_type_): Dependant Feature
            scoring_parameter_for_grid_search_cv (str): Scoring paramater for Grid Search
            base_recall (float, optional): Minimum desired recall. Defaults to 0.9.

        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            BestModel: The best model with its parameters
        """
        try:
            logging.info(f"Initializing all models mentioned in Model.yaml")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized all the models")
            grid_search_best_model_list = self.initialize_best_paramter_search_for_initialized_models(initialized_model_list=
                                                                                                      initialized_model_list, input_feature=X,
                                                                                                      output_feature=y, scoring_parameter=scoring_parameter_for_grid_search_cv)
            best_model = ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_search_best_model_list= grid_search_best_model_list)
            
            return best_model
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e