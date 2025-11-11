from src.mlflow_configurations import load_tokenizer_name,test_model_versioning


## define the model's parameters:
tokenizer_name=load_tokenizer_name()

stage="production"

test_tokenizer_prod=test_model_versioning(name=tokenizer_name,stage=stage)


print(test_tokenizer_prod)