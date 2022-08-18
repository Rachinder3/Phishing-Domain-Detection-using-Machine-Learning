import imp
from phishing_domain_detection.constants import *
from phishing_domain_detection.entity.config_entity import DataTransformationConfig
from phishing_domain_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from phishing_domain_detection.exception import Phishing_Exception
import os, sys
from phishing_domain_detection.logger import logging
import pandas as pd
import numpy as np
import scipy.stats as stat

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from phishing_domain_detection.util.util import *

class box_cox_transformation(BaseEstimator, TransformerMixin):
    
    def __init__(self, features, flag=False):
        
        self.features = features
        self.flag = flag
        logging.info(f"Started applying box cox transformation on:{self.features}")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self,X,y=None):
        #df = X.copy()
        if not self.flag:
            logging.info("Flag false, hence not applying box cox transformation")
            return X
        
        for feature in self.features:
            try:
                # Apply box cox transformations
                X[feature], parameter = stat.boxcox(X[feature])
            except Exception as e:
                raise Phishing_Exception(e,sys) from e
        
        logging.info(f"Applied box cox transformation on: {self.features}")
        return X
            

class DataTransformation:
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation Log Started {'>>'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
        
    def get_data_transformer_object(self):
        try:
            logging.info(f"Generating the Data Transformer Object")
            
            
            
            use_box_cox_or_not = self.data_transformation_config.use_box_cox_transformation
            
            box_cox_object = box_cox_transformation(BOX_COX_FEATURES, use_box_cox_or_not)
            
            pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('box_cox_transformation', box_cox_object),
                ('std_scaler',StandardScaler())
            ])
            
            return pipeline
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def initialize_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            logging.info(f"Obtaining training and testing file path.")
            
            
            logging.info("Loading training and testing data as pandas dataframe.")
            
            train_df = load_data(file_path=self.data_ingestion_artifact.train_file_path, schema_file_path= self.data_validation_artifact.schema_file_path)
            
            test_df = load_data(file_path=self.data_ingestion_artifact.test_file_path, schema_file_path= self.data_validation_artifact.schema_file_path)
            
            schema = read_yaml_file(file_path= self.data_validation_artifact.schema_file_path)
            
            target_column = schema[SCHEMA_TARGET_COLUMN_KEY]
            
            logging.info(f"Splitting input and target from train and test dataframes")
            
            input_feature_train_df, target_feature_train_df = train_df.drop(columns=[target_column], axis = 1), train_df[target_column]
            
            input_feature_test_df, target_feature_test_df = test_df.drop(columns=[target_column], axis = 1), test_df[target_column]
            
            
            logging.info(f"Applying preprocessing on training and testing files")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            
            train_file_name = os.path.basename(self.data_ingestion_artifact.train_file_path) + ".npz"
            test_file_name = os.path.basename(self.data_ingestion_artifact.test_file_path) + ".npz"
            
            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
            
            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(transformed_train_file_path, train_arr)
            save_numpy_array_data(transformed_test_file_path, test_arr)
            
            
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path
            
            logging.info(f"Saving the preprocessing object")
            
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)
            
            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                message=f"Data Transformation Successful.",
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path
            )
            
            logging.info(f"Data Transformation artifact : {data_transformation_artifact}")
            
            return data_transformation_artifact
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'>>'*20} Data Transformation Log Completed {'>>'*20}")