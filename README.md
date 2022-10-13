# Phishing-Domain-Detection-using-Machine-Learning
The aim of this project is to detect phishing domains with the help of Machine Learning

# Jupyter notebbok
### Things to do:
 EDA: use pandas_profiling and self implementation of EDA. Ignoring pandas profiling as features are too many

 Feature Engineering: Use all the methods in playlist (transformations, cardinality, missing values, normalizations etc). 

 Feature Selection: self implementation from playlist
 Before Model Building: get best threshold (work on False Negatives), See effects of clustering

 Model Building: Check with different models


### To install setup.py
python setup.py install


### To stop all jupyter notebooks
jupyter-notebook stop



## Flow

1. Configurations defined in config / config.yaml

2. Entities (structures) desired by different components defined in config_entity

3. In artifact_entity, we will define outputs generated by these components

4. We connect the config_entities(structures) and config.yaml (configurations) using configuration class in config module in phishing_domain_detection package
We populate the config_entities with configuaration.yaml


5. Utils module: Some utility classes

6. Constants: storing all the different constants

7. Build actual code in component

8. Combine all in pipeline



