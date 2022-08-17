from phishing_domain_detection.pipeline.pipeline import Pipeline
from phishing_domain_detection.logger import logging
from phishing_domain_detection.exception import Phishing_Exception


def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(f"{e}")
    

if __name__=='__main__':
    main()
    