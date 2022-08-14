from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
     ['dataset_dir','dataset_name','table_name','top_features','ingested_train_dir','ingested_test_dir'])

DataValidationConfig = namedtuple("DataValidationConfig", ['schema_file_path'])

DataTransformationConfig = namedtuple('DataTransformationConfig',['use_box_cox_transformation',
                                                                  'transformed_train_dir',
                                                                  'transformed_test_dir',
                                                                  'preprocessed_object_file_path'])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ['trained_model_file_path',
                                                       'base_recall',
                                                       'base_precision'])

ModelEvaluationConfig  = namedtuple("ModelEvaluationConfig",["model_evaluation_file_path","time_stamp"])


ModelExportConfig = namedtuple("ModelExportConfig",["export_dir_path"])


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",["artifact_dir"])


