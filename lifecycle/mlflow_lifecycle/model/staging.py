import mlflow
from src.mlflow_configurations import set_mlflow_exp,set_mlflow_host,mlflow_client,load_model_name



## set the experiment name within the workflow first:
set_mlflow_exp()

## set the mlflow local host too
set_mlflow_host()


## create the mlflow client first:
client=mlflow_client()


## create the variables using the imported functions:
model_name=load_model_name()
model_version="1"


# stage the model using the client variable created earlier in the workflow:
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="staging"
)


