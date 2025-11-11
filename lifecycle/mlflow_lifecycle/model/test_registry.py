from src.mlflow_configurations import load_model_name,test_model_registry


## define the model's parameters:
model_name=load_model_name()

model_version="1"

testing=test_model_registry(name=model_name,version=model_version)


print(testing)