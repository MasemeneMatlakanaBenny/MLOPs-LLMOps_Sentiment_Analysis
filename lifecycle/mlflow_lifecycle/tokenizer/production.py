import mlflow
from src.mlflow_configurations import set_mlflow_exp,set_mlflow_host,mlflow_client,load_tokenizer_name



## set the experiment name within the workflow first:
set_mlflow_exp()

## set the mlflow local host too
set_mlflow_host()


## create the mlflow client first:
client=mlflow_client()


## create the variables using the imported functions:
tokenizer_name=load_tokenizer_name()
tokenizer_version="1"


# stage the model using the client variable created earlier in the workflow:
client.transition_model_version_stage(
    name=tokenizer_name,
    version=tokenizer_version,
    stage="production"
)

