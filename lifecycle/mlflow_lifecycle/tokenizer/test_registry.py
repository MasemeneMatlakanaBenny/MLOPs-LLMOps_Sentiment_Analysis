from src.mlflow_configurations import load_tokenizer_name,test_model_registry


## define the model's parameters:
tokenizer_name=load_tokenizer_name()

tokenizer_version="1"

testing=test_model_registry(name=tokenizer_name,version=tokenizer_version)


print(testing)