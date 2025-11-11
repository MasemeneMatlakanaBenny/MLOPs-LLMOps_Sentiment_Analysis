from prefect import task,flow


@task(timeout_seconds=10*60)
def model_serving():

    """
    Task for serving both the model after moving it to the production phase 
    """

    ## import some libraries first:
    import joblib
    import mlflow.pyfunc
    from src.mlflow_configurations import load_model_name
    from src.mlflow_configurations import set_mlflow_exp
    
    ## set the mlflow created experiment inside the mlflow workflow:
    set_mlflow_exp()

    ## define both the tokenizer and model's arguments/parameters:
    model_name=load_model_name()
   

    ## since it has been moved to the production stage:
    stage="production"

    model_uri=f"models:/{model_name}/{stage}"
   

    ## serve the model and tokenizer now:
    model=mlflow.pyfunc.load_model(model_uri=model_uri)
   

    return model

@task(timeout_seconds=10*60)
def tokenizer_serving():

    """
    Task for serving the tokenizer after moving it to the production phase 
    """

    ## import some libraries first:
    import joblib
    import mlflow.pyfunc
    from src.mlflow_configurations import load_tokenizer_name
    from src.mlflow_configurations import set_mlflow_exp
    
    ## set the mlflow created experiment inside the mlflow workflow:
    set_mlflow_exp()

    ## define the tokenizer parameters
    tokenizer_name=load_tokenizer_name()

    ## since it has been moved to the production stage:
    stage="production"

    tokenizer_uri=f"models:/{tokenizer_name}/{stage}"

    ## serve the tokenizer now:

    tokenizer=mlflow.pyfunc.load_model(model_uri=tokenizer_uri)

    return tokenizer



@flow
def serving_flow():
    tokenizer=tokenizer_serving()
    model=model_serving()



