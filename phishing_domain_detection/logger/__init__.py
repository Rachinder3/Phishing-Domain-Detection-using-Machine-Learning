import logging
import datetime
import os

LOG_DIR = "phishing_logs"  # directory where logs will be stored

CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

LOG_FILE_NAME = f"log_{CURRENT_TIMESTAMP}.log"  # creating log file with the help of timestamp

os.makedirs(LOG_DIR,exist_ok=True) # Create log directory, if directory exists, don't create new directory

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME) # path of the file

logging.basicConfig(filename=LOG_FILE_PATH, 
                    filemode='w',
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                    level = logging.INFO)
