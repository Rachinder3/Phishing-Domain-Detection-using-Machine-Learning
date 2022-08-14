import sys, os

from phishing_domain_detection.entity.config_entity import DataIngestionConfig
from phishing_domain_detection.exception import Phishing_Exception
from phishing_domain_detection.logger import logging
from phishing_domain_detection.entity.artifact_entity import *

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import StratifiedShuffleSplit



class sqlite:
    """class to make sqlite queries
    """
    def __init__(self, db_path) -> None:
        """Initializer function for sqlite class

        Args:
            db_path (_type_): path where sqlite database is kept
        """
        self.db_path = db_path
        
    def execute_query_without_commit(self, query):
        """ Executes the given query on the database. This function doesn't perform the commit.
        So Read queries can only be executed      

        Args:
            query (_type_): Query to execute
        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            _type_: Result of executed query
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            res = cur.execute(query)
            return res
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    

class DataIngestion:
    
    def __init__(self, data_ingestion_config : DataIngestionConfig) -> None:
        """Initializer function for DataIngestion class

        Args:
            data_ingestion_config (DataIngestionConfig): Data Ingestion Config coming from configuration module

        Raises:
            Phishing_Exception: Custom Exception
        """
        try:
            logging.info(f"{'>>'*20} Data Ingestion Log Started {'>>'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    
    def read_data_from_db(self) -> pd.DataFrame:
        """ Reads the data from database, ingests it and returns a dataframe.

        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            pd.DataFrame: Ingested Data Frame
        """
        try:
            
            
            logging.info(f"Performing the query on the table")
             # Getting the directory where dataset is kept
            dataset_dir = self.data_ingestion_config.dataset_dir
            
            # Getting the name of the dataset
            dataset_name = self.data_ingestion_config.dataset_name
            
            # Creating path of the dataset
            dataset_path = os.path.join(dataset_dir, dataset_name)
            
            # getting the table name
            table_name = self.data_ingestion_config.table_name
            
            
            # top features 
            top_features = self.data_ingestion_config.top_features
            
            
            
            ### features to query
            features_to_query = str(top_features).replace("[","").replace("'",'''"''').replace("]","")
            
            ## Building the query
            query = f"select {features_to_query} from {table_name}"
            
            ## Execute the query and get the results
            sqlite_obj = sqlite(db_path=dataset_path)
            res = sqlite_obj.execute_query_without_commit(query=query)
            
            logging.info(f"Table extractded")
            ## Converting the result to dataframe and returning
            records = []
            for i in res:
                records.append(i)
                
            ingested_df = pd.DataFrame(records, columns=top_features)
            return ingested_df
            
            
            
            
            
        except Exception as e:
            raise Phishing_Exception(e,sys) from e
    
    def split_data_as_train_test(self, ingested_df : pd.DataFrame) -> DataIngestionArtifact:
        """ Performs Stratified Shuffle split to generate Train and Test Dataset.
        Stratified Split performed so that distribution in Train and Test Dataset is same and real
        life situation is replicated

        Args:
            ingested_df (pd.DataFrame): The Ingested Dataframe

        Raises:
            Phishing_Exception: Custom Exception

        Returns:
            DataIngestionArtifact: Artifact Generated from Data Ingestion Component 
        """
        
        try:
           
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            logging.info(f"Splitting the data into training and testing")

            strat_train_set = None
            
            strat_test_set = None
            
            table_name = self.data_ingestion_config.table_name
            
            for train_index, test_index in split.split(ingested_df, ingested_df['phishing']):
                strat_train_set = ingested_df.loc[train_index]
                strat_test_set = ingested_df.loc[test_index]
            
            
            training_file_path = os.path.join(
                self.data_ingestion_config.ingested_train_dir,
                table_name
            )
            
            testing_file_path = os.path.join(
                self.data_ingestion_config.ingested_test_dir,
                table_name
            )
            
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting training dataset to file: {training_file_path}.csv")
                strat_train_set.to_csv(training_file_path+".csv", index= False)
            
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting test data to file : {testing_file_path}.csv")
                strat_test_set.to_csv(testing_file_path+".csv", index = False)
        
            data_ingestion_artifact = DataIngestionArtifact(
            train_file_path=training_file_path,
            test_file_path=testing_file_path,
            is_ingested=True,
            message=f"Data Ingestion Completed successfully"
            )
            
            logging.info(f"Data Ingestion Artifact: [ {data_ingestion_artifact} ] ")
            return data_ingestion_artifact                    
        except Exception as e:
            raise Phishing_Exception(e,sys) from e    
    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            ingested_df = self.read_data_from_db()
            return self.split_data_as_train_test(ingested_df=ingested_df)
        except Exception as e:
            raise Phishing_Exception(e, sys) from e
    
    def __del__(self):
        logging.info(f"{'>>'*20} Data Ingestion Log Completed {'>>'*20}")
        