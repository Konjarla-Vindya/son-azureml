print("Entered into .PY file")
import os 
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import mlflow
import pandas as pd
import numpy as np
from datasets import load_dataset
import evaluate
from azureml.core import Workspace
from transformers import AutoModelForSequenceClassification,AutoTokenizer,TrainingArguments,Trainer
import pickle
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azure.ai.ml import MLClient
import mlflow
from tensorflow.keras import Model
from azure.ai.ml.entities import AmlCompute
import time

print("imported packages creating experiment")
# subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
# resource_group = 'sonata-test-rg'
# workspace_name = 'sonata-test-ws'

# credential = AzureCliCredential()
# ws = Workspace(subscription_id, resource_group, workspace_name)

# workspace_ml_client = MLClient(
#         credential, subscription_id, resource_group, ws
#     )

# # store the mlflow results in azure ml workspace
# mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

EXPERIMENT_NAME = "Distlbert"
mlflow.set_experiment(EXPERIMENT_NAME)
print("Experiment created")
import mlflow.pytorch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
print("Imported tokenisation packages")
checkpoint = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
print("Done pretraining")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("Done tokenisation")
raw_ds= load_dataset("glue", "mrpc")
raw_ds['train'] = raw_ds['train'].shuffle().select(range(100))
raw_ds['test']= raw_ds['test'].shuffle().select(range(20))
raw_ds['validation']= raw_ds['validation'].shuffle().select(range(20))
metric = evaluate.load("glue", "mrpc")
print("Loaded dataset")
dataset = raw_ds.map(
    lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True),
    batched=True,
)
dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
dataset = dataset.rename_column("label", "labels")
dataset = dataset.with_format("torch")
print("Preparing training args")
trainer_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")


def compute_metrics(eval_preds: EvalPrediction):
    x, y = eval_preds
    preds = np.argmax(x, -1)
    return metric.compute(predictions=preds, references=y)

mlflow.transformers.save_model(
    transformers_model={"model": model, "tokenizer": tokenizer },
    path="./distlbertsavedmodel"
    # signature=signature,
    # input_example=data,
)

with mlflow.start_run():
    # registered_model_name="Albert"
    model_local_path = os.path.abspath("./distlbertsavedmodel")
    mlflow.register_model(f"file://{model_local_path}", "distlbertModel")
	
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=trainer_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
    # test_dataset=dataset["test"]
)

result = trainer.train()

#retrieve trained model

trained_model = trainer.model
trained_model.config

mlflow.end_run()	
print("Congo")