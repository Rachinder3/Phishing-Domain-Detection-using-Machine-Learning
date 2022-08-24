from collections import namedtuple
from unicodedata import name

DataIngestionArtifact = namedtuple("DataIngestionArtifact",[
    "train_file_path",
    "test_file_path",
    "is_ingested",
    "message"
])

DataValidationArtifact = namedtuple("DataValidationArtifact",[
    "schema_file_path",
    "report_file_path",
    "report_page_file_path",
    "is_validated",
    "message"
])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",[
    "is_transformed","message","transformed_train_file_path",
    "transformed_test_file_path","preprocessed_object_file_path"
])


ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",[
    "is_trained","message","trained_model_file_path",
    "train_recall","test_recall","train_precision","test_precision","model_f1", "custom_threshold",
    "model_recall", "model_precision", "base_recall","base_precision"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact",[
    "is_model_accepted",
    "evaluated_model_path"
])

