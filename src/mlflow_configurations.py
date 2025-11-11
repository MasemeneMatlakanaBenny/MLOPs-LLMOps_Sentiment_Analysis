## create the configurations for mlflow for reusability purposes:
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException


host="http://127.0.0.1:5000"

exp_name="sentiment movie classification"

exp_description="Performing sentiment analysis on the movie review dataset"

tags={
    "project_name":"sentimen_analysis",
    "team":"AI/ML Team",
    "project_date":"11 November 2025",
    "mlflow.note.content":exp_description
}

model_name="distil_model"


tokenizer_name="distil_tokenizer"

# this is where we now create our libraries that we will reuse in the entire workflow of the project:

## this is a function that will be used to load the local host /server of mlflow
def load_host(host=host):
    return host

## a function for loading the experiment name in mlflow:
def load_exp_name(name=exp_name):
    return name

## the function for setting the local host within the mlflow workflow:
def mlflow_client(server=host):
    client=MlflowClient(tracking_uri=server) 

    return client


## the function for setting the experiment name within the mlflow workflow:
def set_mlflow_exp(name=exp_name):
    return mlflow.set_experiment(name=name)

## the function for setting the local host within the mlflow workflow:
def set_mlflow_host(server=host):
    return mlflow.set_tracking_uri(uri=server)


## two functions for loading the model and tokenizer
### load the model name
def load_model_name():
    return model_name

### load the tokenizer name
def load_tokenizer_name():
    return tokenizer_name


## function for testing model registry
def test_model_registry(name, version):
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, version=version)
        print(f"Model '{name}' version {version} exists")
    except RestException:
        print(f" Model '{name}' version {version} not found")
        

## a function for testing model versioning or stages per model
def test_model_versioning(name, stage):
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, stage=stage)
        print(f"Model '{name}' at {stage} exists")
    except RestException:
        print(f"Model '{name}' at {stage} not found")