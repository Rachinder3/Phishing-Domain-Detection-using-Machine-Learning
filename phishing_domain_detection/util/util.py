# set of utility classes and functions

import yaml
from phishing_domain_detection.exception import Phishing_Exception
import os, sys

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
    
    