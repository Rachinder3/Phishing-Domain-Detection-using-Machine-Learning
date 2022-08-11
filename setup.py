from typing import List
from setuptools import setup, find_packages

# Declaring variables for setup function
PROJECT_NAME = "phishing_domain_detection"
PROJECT_VERSION = "0.0.3"
PROJECT_AUTHOR = "Rachinder Singh"
PROJECT_AUTHOR_EMAIL = "rachindersingh@gmail.com" 
PROJECT_DESCRIPTION = "The project builds a smart system capable of detecting a particular domain as phishing or non phishing"
PROJECT_PACKAGES = ["phishing_domain_detection"]
REQUIREMENTS_FILE_NAME = "requirements.txt"

## Function to get all the requirements listed in requirements.txt
def get_requirements_list() -> List[str]:
    """ This function reads the requirments.txt and returns the list of requirements listed
    within it.

    Returns:
        List[str]: List of requirements listed in requirements.txt
    """
    requirements_list = None
    with open(REQUIREMENTS_FILE_NAME) as file_obj:
        requirements_list = file_obj.readlines()
    
    requirements_list = [requirement.replace("\n","") for requirement in requirements_list]
    
    return requirements_list
    

# Executing the setup function
setup(
    name= PROJECT_NAME,
    version= PROJECT_VERSION,
    author=PROJECT_AUTHOR,
    author_email=PROJECT_AUTHOR_EMAIL,
    description= PROJECT_DESCRIPTION,
    packages= find_packages(), ## will find all the custom packages present in the root directory on its own
    install_requires = get_requirements_list()
)

    