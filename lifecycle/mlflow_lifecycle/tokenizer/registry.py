
import mlflow
from src.mlflow_configurations import set_mlflow_exp,set_mlflow_host
from src.mlflow_configurations import load_exp_name,load_tokenizer_name,load_tags,load_host
from src.configurations import load_tokenizer


## set the experiment name within the workflow first:
set_mlflow_exp()

## set the mlflow local host too
set_mlflow_host()


## create the variables using the imported functions:

tokenizer_name=load_tokenizer_name()


checkpoint="distilbert-cased-finetuned-sst2-english"

## load the model:
tokenizer=load_tokenizer(checkpoint=checkpoint)

## register the model on mlflow:
### define the name of the run for the model registry first:
run_name="tokenizer_registry_run"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.pyfunc.log_model(tokenizer,registered_model_name=tokenizer_name)
    
