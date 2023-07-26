print("execution started")

import mlflow
import transformers


modelname = 'gpt2-medium'
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

print("before tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(modelname)
print("after tokenizer model")
model = GPT2LMHeadModel.from_pretrained(modelname)
print("after automodel")



basemodel = mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer },
        task="text-generation",
        artifact_path="models/basemodel",
        input_example=["base model loaded directly from transformer's"],
        registered_model_name="Gpt-2-medium-base-model"
        )
print("registered the model")


mlflow.transformers.save_model(
    transformers_model={"model": model, "tokenizer": tokenizer },
    path="transformersmodel"
)

mlmodel = mlflow.transformers.load_model(
    "transformersmodel"
)



from datasets import load_dataset
traindataset = load_dataset("cnn_dailymail","3.0.0", split="train").shuffle().select(range(1000))
testdataset = load_dataset("cnn_dailymail", "3.0.0", split="test").shuffle().select(range(500))

def tokenizer_function(examples):
        return tokenizer(examples["article"],text_target=examples["highlights"],truncation=True,padding=True,max_length=128,return_tensors="pt")

tokenizer.pad_token = tokenizer.eos_token
traindataset = traindataset.map(tokenizer_function,batched=True)
testdataset = testdataset.map(tokenizer_function,batched=True)

tokenizer.pad_token = tokenizer.eos_token
traindataset = traindataset.map(tokenizer_function,batched=True)
testdataset = testdataset.map(tokenizer_function,batched=True)

traindataset = traindataset.remove_columns(["article","highlights","id"])
testdataset = testdataset.remove_columns(["article","highlights","id"])

import evaluate
metrics = evaluate.load("perplexity")

import numpy as np
def compute_metrics(eval_period):
        print(eval_period) 
        logits, labels = eval_period
        predictions = np.argmax(logits,x_axis = 1)
        return metrics.compute(predictions = predictions, references = labels)


import os
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

from transformers import TrainingArguments,Trainer

Training_args = TrainingArguments(
        num_train_epochs = 1,
        output_dir= "output",
        save_strategy="epoch"
    )

trainer = Trainer(
        model = mlmodel.model,
        tokenizer=tokenizer,
        args = Training_args,
        compute_metrics=compute_metrics,
        train_dataset = traindataset,
        eval_dataset = testdataset,
        
    )

result = trainer.train()

print(result)
# trainedmodel = mlflow.transformers.log_model(
#         transformers_model={"model": trainer.model, "tokenizer": tokenizer },
#         task="text-generation",
#         artifact_path="models/trainedmodel",
#         input_example=["base model loaded directly from transformer's"],
#         registered_model_name="Gpt-2-medium-base-model"
#         )

# mlflow.end_run()

# print("trainer model got logged successfully")

