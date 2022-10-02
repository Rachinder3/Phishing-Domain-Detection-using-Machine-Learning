
from flask import Flask, jsonify, request


import os, sys
import json

from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception
import os,sys
from phishing_domain_detection.util.util import check
from phishing_domain_detection.entity.phishing_estimator import PhishingEstimator



# initializations
app = Flask(__name__)
model_directory = "saved_models"




# views
@app.route("/",methods=['GET','POST'])
def index():
    return f"<h1> Phishing Domain Detection Project {check()}</h1>"
    
    
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
        print(logging.info(exception.error_message))
    
    
if __name__=='__main__':
    app.run()
    