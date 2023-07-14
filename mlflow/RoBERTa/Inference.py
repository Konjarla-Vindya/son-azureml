from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
)
from azure.ai.ml.entities import AmlCompute
import time

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

workspace_ml_client = MLClient(
    credential,
    subscription_id="80c77c76-74ba-4c8c-8229-4c3b2957990c",
    resource_group_name="sonata-test-rg",
    workspace_name="sonata-test-ws",
)
# the models, fine tuning pipelines and environments are available in the AzureML system registry, "azureml"
registry_ml_client = MLClient(credential, registry_name="azureml")
model_name = "Jean-Baptiste-camembert-ner"
version_list = list(registry_ml_client.models.list(model_name))
if len(version_list) == 0:
    print("Model not found in registry")
else:
    model_version = version_list[0].version
    foundation_model = registry_ml_client.models.get(model_name, model_version)
    print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
            foundation_model.name, foundation_model.version, foundation_model.id
        )
    )
# Download a small sample of the dataset into the ./polyglot_ner-dataset directory
%run ./download-dataset.py --download_dir ./polyglot_ner-dataset
# load the ./polyglot_ner/train.jsonl file into a pandas dataframe and show the first 5 rows
import pandas as pd

pd.set_option(
    "display.max_colwidth", 0
)  # set the max column width to 0 to display the full text
train_df = pd.read_json("./polyglot_ner-dataset/train.jsonl", lines=True)

train_df.drop(columns=["words", "id", "lang"], inplace=True)
train_df.rename(columns={"ner": "ground_truth_labels"}, inplace=True)

train_df = train_df[["text", "ground_truth_labels"]]

train_df.head()
# create a deployment
demo_deployment = ManagedOnlineDeployment(
    name="demo",
    endpoint_name=online_endpoint_name,
    model=foundation_model.id,
    instance_type="Standard_DS2_v2",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=60000,
    ),
)
workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
endpoint.traffic = {"demo": 100}
workspace_ml_client.begin_create_or_update(endpoint).result()
import json
import os

# pick 1 random row
sample_df = train_df.sample(1)
# create a json object with the key as "inputs" and value as a list of values from the en column of the sample_df dataframe
sample_json = {"inputs": sample_df["text"].tolist()}
# save the json object to a file named sample_score.json in the ./polyglot_ner-dataset folder
test_json = {"inputs": {"input_string": sample_df["text"].tolist()}}
# save the json object to a file named sample_score.json in the ./polyglot_ner-dataset folder
with open(os.path.join(".", "polyglot_ner-dataset", "sample_score.json"), "w") as f:
    json.dump(test_json, f)
sample_df.head()
# score the sample_score.json file using the online endpoint with the azureml endpoint invoke method
response = workspace_ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="demo",
    request_file="./polyglot_ner-dataset/sample_score.json",
)
print("raw response: \n", response, "\n")
# convert the json response to a pandas dataframe
response_df = pd.read_json(response)
response_df.head()
# compare the predicted labels with the actual labels
predicted_labels = response_df[0][0]
compare_df = pd.DataFrame(
    {
        "ground_truth_labels": sample_df["ground_truth_labels"].tolist(),
        "predicted_labels": [predicted_labels],
    }
)
compare_df.head()
workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait()