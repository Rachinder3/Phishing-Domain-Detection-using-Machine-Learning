# set of utility classes and functions

import imp
import yaml
from phishing_domain_detection.exception import Phishing_Exception
import os, sys
import pandas as pd
from phishing_domain_detection.constants import *
import joblib
import numpy as np

def read_yaml_file(file_path:str)->dict:
    """Reads the yaml file for which path is provided
    and returns the dictionary representation of yaml file

    Args:
        file_path (str): path for the yaml file

    Raises:
        e: Exception object
    Returns:
        dict: The dictionary representation of yaml file
    """
    
    
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Phishing_Exception(e, sys) from e
    
 
  
def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    """Loads the data kept in file_path, typecasts the columns of the file to the format as mentioned in schema

    Args:
        file_path (str): Path of the file which we are loading
        schema_file_path (str): Path of the schema file

    Raises:
        Phishing_Exception: Custom Exception

    Returns:
        pd.DataFrame: Loaded and typecasted data
    """
    try:
        
        dataset_schema = read_yaml_file(schema_file_path)
        
        schema = dataset_schema[SCHEMA_COLUMNS_KEY]
        schema[dataset_schema[SCHEMA_TARGET_COLUMN_KEY]] = dataset_schema[SCHEMA_TARGET_COLUMN_TYPE_KEY] ## Adding Phishing key in the schema as well
        print(schema)
        
        dataframe = pd.read_csv(file_path)
        
        error_message = "" # Helps to ensure that typecasting is proper, if not, we will shut down the pipeline
        
        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_message = f"{error_message} \nColumn: [{column}] is not in the schema."       
                break
        
        if len(error_message) > 0:
            raise Exception(error_message)
        
        return dataframe
        
        
    except Exception as e:
        raise Phishing_Exception(e,sys) from e
    
def save_object(file_path:str, obj) -> None:
    """Dumps the object in the given file path using joblib library

    Args:
        file_path (str): Path where object is supposed to be dumped
        obj (_type_): Object to be dumped

    Raises:
        Phishing_Exception: Custom Exception
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        joblib.dump(obj,file_path)
    except Exception as e:
        raise Phishing_Exception(e,sys) from e
    
def load_object(file_path:str):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise Phishing_Exception(e,sys) from e
    
    
def save_numpy_array_data(file_path:str, array: np.array):
    """Saves the given numpy array into the file path

    Args:
        file_path (str): Path to store the data
        array (np.array): Array to be stored

    Raises:
        Phishing_Exception: Custom Exception
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            np.save(file_obj, array)
        
    except Exception as e:
        raise Phishing_Exception(e,sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """Loads the numpy array from given file

    Args:
        file_path (str): String location of file to load

    Raises:
        Exception: Custom Exception
        
    Returns:
        np.array: Numpy array loaded
    """
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise Exception(e,sys) from e