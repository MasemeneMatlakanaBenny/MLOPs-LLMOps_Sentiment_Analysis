from src.mlflow_configurations import load_model_name,test_model_versioning


## define the model's parameters:
model_name=load_model_name()

stage="production"

test_model_prod=test_model_versioning(name=model_name,stage=stage)


print(test_model_prod)