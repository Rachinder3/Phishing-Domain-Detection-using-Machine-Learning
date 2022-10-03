
from genericpath import isfile
from multiprocessing import context
from pyexpat.errors import messages
from flask import Flask, jsonify, request, render_template, send_file, abort

import os, sys
import json


from phishing_domain_detection.exception import Phishing_Exception
import os,sys
from phishing_domain_detection.util.util import check
from phishing_domain_detection.entity.phishing_estimator import PhishingEstimator
from phishing_domain_detection.pipeline.pipeline import Pipeline
from phishing_domain_detection.constants import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME

import pandas as pd
# initializations
app = Flask(__name__)

ROOT_DIR = os.getcwd()
model_directory = "saved_models"
pipeline_folder = "phishing_domain_detection"
artifacts_folder = "artifacts"

LOG_FOLDER_NAME = "phishing_logs"
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)

experiment_file_path = os.path.join(pipeline_folder,artifacts_folder, EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
rows_to_return = 5


## utility function
def generate_experiment_history_df():
    try:
        #print(experiment_file_path)
        if not os.path.exists(experiment_file_path):
            return pd.DataFrame()
        experiment_df = pd.read_csv(experiment_file_path)
        
        if 'experiment_file_path' in experiment_df.columns:
            experiment_df.drop('experiment_file_path', axis = 1, inplace = True)
        
        if 'initialization_timestamp' in experiment_df.columns:
            experiment_df.drop('initialization_timestamp', axis = 1, inplace = True)
            
        
        experiment_df  = experiment_df.iloc[-1:(-1*rows_to_return)-1:-1]
        #print(experiment_df)
        
        return experiment_df
    except Exception as e:
        print(str(e))
    
    
    


# views
@app.route("/",methods=['GET','POST'])
def index():
    return render_template("index.html")
    
    
@app.route("/predictions_via_api", methods = ['GET','POST'])
def predictions_via_api():
    try:
        if request.method == 'POST':
            
            
            ######### Reading the data
            urls = request.json["urls"]
            
            
            ### url_list can be string or list of strings, so convert single string to list of strings
            if type(urls) == str: 
                urls = [urls]
            
            
            
            
            ## checking if no model in production
            if not (os.path.exists(model_directory)) or len(os.listdir(model_directory))<=0 :
               
                return jsonify({
                    "response":"No model in prodiction, run the training pipeline atleast once"
                })

            

            ### predictions
            pe = PhishingEstimator(model_directory) ### creating object of phishing estimator
            
            results = pe.predict(urls)
        
            results = list(results)
            
            for index, result in enumerate(results):
                if result == 0:
                    results[index] = "Not Phishing"
                else:
                    results[index] = "Phishing"
              
                  
            
            return jsonify({
                "results": dict(zip(urls, results))
                })
            
            
           
            
    except Exception as e:
        exception = Phishing_Exception(e,sys) 
        print(exception.error_message)
    
@app.route("/predictions", methods=['GET','POST'])
def predictions_via_web():
    
    try:
        context = None
    
    
        if request.method == 'POST':
            urls = request.form['urls']

            if(urls == ''): ## if no url passed then don't do anything
                return render_template("predictions.html", context = context)
            
            urls = urls.split("\r\n")
            
            
            urls = [url.strip() for url in urls]
            
           
                
            
            print(urls)
            
            if not (os.path.exists(model_directory)) or len(os.listdir(model_directory))<=0 :
                context = -1
            else:
                pe = PhishingEstimator(model_directory) ### creating object of phishing estimator
                
                results = pe.predict(urls)
                
                results = list(results)
                
                print(results)
                for index, result in enumerate(results):
                    if result == 0:
                        results[index] = "Not Phishing"
                    else:
                        results[index] = "Phishing"
                
                context = dict(zip(urls, results))
                
                print(context)
            return render_template("predictions.html", context = context)
        return render_template("predictions.html", context = context)
            
            
    except Exception as e:
        exception = Phishing_Exception(e,sys)
        print(exception.error_message)
    
@app.route("/train", methods = ["GET","POST"])
def train():
    
    try:
        context = None
        
        pipeline = Pipeline()
        if request.method == "POST":
            if Pipeline.experiment.running_status:
                ## If pipeline already running, then dont start another pipeline
                context = dict()
                context["state"] = 2
                message = "Pipeline is already running. Please wait some time before triggering another pipeline."
                context["message"] = message
            else:
                context = dict()
                context["state"] = 1
                pipeline.start()
                message = "Pipeline has been triggered."
                context["message"] = message
            
            context["experiment"] = generate_experiment_history_df()

            return render_template("train.html", context = context)
        return render_template("train.html", context=context)
        
    except Exception as e:
        exception = Phishing_Exception(e,sys)
        print(exception.error_message)


@app.route("/experiment_history", methods = ["GET","POST"])
def experiment_history():
    try:
        context = dict()
        
        context["experiment"] = generate_experiment_history_df()
        
        return render_template("experiment_history.html", context = context)
    except Exception as e:
        exception = Phishing_Exception(e,sys)
        print(exception.error_message)
           
    
@app.route("/logs", defaults = {'req_path': f"{LOG_FOLDER_NAME}"})
@app.route(f"/logs/<path:req_path>")
def render_log_dir(req_path):
    try:
        os.makedirs(LOG_FOLDER_NAME, exist_ok=True)  ## If the log directory doesn't exist, this will create the directory
        
        print(f"req_path: {req_path}")
        abs_path = os.path.join(req_path)
        print(f"absolute path : {abs_path}")
        
        if not os.path.exists(abs_path):
            return abort(404)
        
        
        
        if os.path.isfile(abs_path):
            return send_file(abs_path)
        
        
        ## Getting directory contents
        files = {os.path.join(abs_path,file): file for file in os.listdir(abs_path)[::-1]}
        
        
        result = {
            "files" : files,
            "parent_folder": os.path.dirname(abs_path),
            "parent_label": abs_path
        }
        
        return render_template("logs.html", result =result)
        
    except Exception as e:
        exception = Phishing_Exception(e,sys)
        print(exception.error_message)


@app.route("/saved_models", defaults = {'req_path':f"{model_directory}"})
@app.route("/saved_models/<path:req_path>")
def saved_models(req_path):
    try:
        ### If no folder exists, create an empty model
        os.makedirs("saved_models", exist_ok=True)
        
        ## Joining the base and requested path
        print(f"requested_path:{req_path}")
        
        abs_path = os.path.join(req_path)
        
        print("abs_path")
        
        ## Return 404 if path doesn't exist
        if not os.path.exists(abs_path):
            return abort(404)
        
        ## Check if path is file and serve
        if os.path.isfile(abs_path):
            return send_file(abs_path)
        
        ## Show directory contents  
        files = {os.path.join(abs_path,file):file for file in os.listdir(abs_path)[::-1]}
        
        result = {
            "files": files,
            "parent_folder": os.path.dirname(abs_path),
            "parent_label": abs_path
        }
        
        return render_template("saved_models.html",result = result)
        
    except Exception as e:
        exception = Phishing_Exception(e,sys)
        print(exception.error_message)
    

    
    
    
if __name__=='__main__':
    app.run()
    #generate_experiment_history_dict()
    