from flask import Flask
import phishing_domain_detection
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception
import os,sys
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    try:
        logging.info('We are testing logging module')
        raise Exception("We are testing exception class")
        return "<h1> Building phishing project </h1>"
    except Exception as e:
        phishing_exception = Phishing_Exception(e,sys)
        #print(phishing_exception.error_message)
        logging.info(phishing_exception.error_message)
    return "<h1> Phishing Domain Detection Project</h1>"
    
if __name__=='__main__':
    app.run()
    