import mlflow
from src.mlflow_configurations import set_mlflow_exp,set_mlflow_host
from src.mlflow_configurations import load_exp_name,load_model_name,load_tags,load_host
from src.configurations import load_model


## set the experiment name within the workflow first:
set_mlflow_exp()

## set the mlflow local host too
set_mlflow_host()


## create the variables using the imported functions:
host=load_host()

exp_name=load_exp_name()

model_name=load_model_name()

tags=load_tags()


checkpoint="distilbert-cased-finetuned-sst2-english"

## load the model:
model=load_model(checkpoint=checkpoint)

## register the model on mlflow:
### define the name of the run for the model registry first:
run_name="model_registry_run"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.transformers.log_model(transformers_model=model,
                                  registered_model_name=model_name)
    

