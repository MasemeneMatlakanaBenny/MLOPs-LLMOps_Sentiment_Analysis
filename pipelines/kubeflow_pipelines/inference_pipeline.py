## Create the pipeline that can be used to generate the predictions:

## import the libraries first:
from kfp import dsl,compiler
from kfp.dsl import pipeline,component
from src.configurations import prediction_pipeline
from serving import serving_pipeline
from typing import List

@component
def inference_component(input_data:List)->List:
    """
    The component for handling incoming data
    """
    model,tokenizer=serving_pipeline()

    ##now load the prediction pipeline:
    pred_probs=prediction_pipeline(text=input_data,
                                      model=model,
                                      tokenizer=tokenizer)
    return pred_probs


@component
def classifier_component(predicted_probs:List)->List:

    import pandas as pd
    import numpy as np

    pred_df=predicted_probs.to_dataframe()

    pred_df['scores']=np.where(pred_df['probs']>=0.5,1,0)

    return pred_df

@pipeline(
    name="inferencing pipeline",
    description="Used to handle incoming data"
)
def inf_pipeline():
    pred_probs=inference_component()

    classified=classifier_component(predicted_probs=pred_probs)

    return classified


compiler.Compiler().compile(
    pipelien_func=inf_pipeline,
    package_path="pipelines/kubeflow_pipelines/inf_pipe.yaml"
)