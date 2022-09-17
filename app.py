from flask import Flask
import phishing_domain_detection
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception
import os,sys
from phishing_domain_detection.util.util import check
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    return f"<h1> Phishing Domain Detection Project {check()}</h1>"
    
if __name__=='__main__':
    app.run()
    