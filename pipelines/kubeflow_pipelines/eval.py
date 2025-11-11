## import libs first:
from kfp import dsl,compiler
from kfp.dsl import pipeline,component


##create a pipeline for evaluating the model's predicitions:
### start by first creating the component of the pipeline:
@component
def eval_component(input_data,
                   y_true):
    """
    A function for evaluating the model's performance

    """
    from src.configurations import model_metrics
    from inference_pipeline import inf_pipeline

    y_preds=inf_pipeline(input_data=input_data)

    metrics=model_metrics(y_true=y_true,y_preds=y_preds)

    return metrics


@pipeline(
    name="evaluation_ml_pipeline",
    description="pipeline for evaluating the model's performance"
)
def evaluation_pipeline():

    component_metrics=eval_component()

    return component_metrics



compiler.Compiler().compile(
    pipeline_func=evaluation_pipeline,
    package_path="pipelines/kubeflow_pipelines/evaluation_pipeline.yaml"
)


