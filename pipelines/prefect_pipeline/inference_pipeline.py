## Create the pipeline that can be used to generate the predictions:

## import the libraries first:
from prefect import task,flow
from src.configurations import prediction_pipeline
from typing import List

@task
def inference_task(input_data:List)->List:
    """
    The component for handling incoming data
    """
    from serving_pipeline import tokenizer_serving,model_serving
    model=model_serving()

    tokenizer=tokenizer_serving()

    ##now load the prediction pipeline:
    pred_probs=prediction_pipeline(text=input_data,
                                      model=model,
                                      tokenizer=tokenizer)
    return pred_probs

@task
def classifier_task(predicted_probs:List)->List:

    import pandas as pd
    import numpy as np

    pred_df=predicted_probs.to_dataframe()

    pred_df['scores']=np.where(pred_df['probs']>=0.5,1,0)

    return pred_df


@flow
def inf_workflow():
    probs=inference_task()

    predictions=classifier_task()

    return predictions