from kfp import dsl,compiler
from kfp.dsl import pipeline,component,Input,Output,Artifact

# create the component for serving the pipeline:
@component
def serving_component(model_artifact:Output[Artifact],
                      tokenizer_artifact:Output[Artifact]):
    """
    Component for loading both the tokenizer and model from mlflow productions .
    Both the model and tokenizer will be used to generate predictions
    """
    import joblib
    import mlflow.pyfunc
    from src.mlflow_configurations import load_model_name,load_tokenizer_name
    from src.mlflow_configurations import set_mlflow_exp

    ## set the mlflow created experiment inside the mlflow workflow:
    set_mlflow_exp()

    ## define both the tokenizer and model's arguments/parameters:
    model_name=load_model_name()
    tokenizer_name=load_tokenizer_name()

    ## since they both have been moved to the production stage:
    stage="production"

    model_uri=f"models:/{model_name}/{stage}"
    tokenizer_uri=f"models:/{tokenizer_name}/{stage}"

    ## serve the model and tokenizer now:
    model=mlflow.pyfunc.load_model(model_uri=model_uri)
    tokenizer=mlflow.pyfunc.load_model(model_uri=tokenizer_uri)

    ## save both the model and tokenizer:
    joblib.dump(model,model_artifact.path)
    joblib.dump(tokenizer,tokenizer_artifact.path)



## create the pipeline now:
@pipeline(
    name="serving_pipeline",
    description="Pipeline for serving the model and tokenizer after moving to the production phase"
)
def serving_pipeline():

    served_artifacts=serving_component()

    model=served_artifacts.outputs['model_artifact']

    tokenizer=served_artifacts['tokenizer_artifact']

    return model,tokenizer


## compile the pipeline:
compiler.Compiler().compile(
    pipeline_func=serving_pipeline,
    package_path="pipelines/kubeflow_pipelines/model_tokenizer_serving_pipeline.yaml"
)