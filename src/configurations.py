# import libs first:
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

## create some configurations for the workflow:

def load_tokenizer(checkpoint)->AutoTokenizer:
    """This is a function that will be used to reload the tokenizer again and again"""
    
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)

    return tokenizer


## create the function for loading the model:
def load_model(checkpoint)->AutoModelForSequenceClassification:
    """This is a function that will be used to get the model"""


    #load the model:
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

    ## return the model loaded:
    return model


## create the pipeline for prediction:
def prediction_pipeline(text,model:AutoModelForSequenceClassification,tokenizer:AutoTokenizer):
    """
    The pipeline for text classification  
    """
    # get the inputs first:
    inputs=tokenizer(text,return_tensors="pt")

    outputs=model(**inputs)

    ## get the logits after computing for the outputs:
    logits=outputs.logits

    ## get the predictions:
    predictions=torch.nn.functional.softmax(logits)


    ## this will return the tensor with the predicted probability:
    return predictions[1]


import torch

def tensor_readable(t: torch.Tensor, decimals=6):
    """
    Convert a tensor to readable decimals.
    """
    return t.detach().cpu().numpy().round(decimals).tolist()




## the function for getting the model evaluation:
def model_metrics(y_true,y_preds):
    from sklearn.metrics import cohen_kappa_score,matthews_corrcoef,accuracy_score,confusion_matrix

    kappa_score=cohen_kappa_score(y_true,y_preds)
    mat_score=matthews_corrcoef(y_true,y_preds)
    acc_score=accuracy_score(y_true,y_preds)

    metrics={
        "kappa_score":kappa_score,
        "matt_corr":mat_score,
        "acc_score":acc_score
    }

    return metrics


def model_conf_mat(y_true,y_preds):
    from sklearn.metrics import confusion_matrix

    conf_mat=confusion_matrix(y_true,y_preds)

    return conf_mat

def save_model_metrics(metrics):
    import joblib
    joblib.dump(metrics,"model_metrics/metrics.pkl")

