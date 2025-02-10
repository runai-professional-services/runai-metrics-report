import random
import string
from runai.client import RunaiClient


def random_string(length: int):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length)).lower()


client = RunaiClient(
    client_id="api",
    client_secret="7GlUoP8mWfndgNSBS6RH00eTQ761mrKT",
    runai_base_url="https://runai.dilerous.cloud",
    cluster_id="ce75a5f5-0fc5-48fb-8d2e-68236fbcf417",
)
q_t = {
        "image": "gcr.io/run-ai-demo/quickstart-demo",
        "imagePullPolicy": "Always",
        "nodePools": ["default"],
        "compute": {
            "cpuCoreRequest": 0.2,
            "cpuMemoryRequest": "10M",
            "gpuDevicesRequest": 1,
        },
        "backoffLimit": 6,
        "priorityClass": "train",
    }
for i in range(10):
    print(client.training.create(
        training_name="quickstart-"+random_string(4),
        use_given_name_as_prefix=False,
        project_id="4500001",
        cluster_id="ce75a5f5-0fc5-48fb-8d2e-68236fbcf417",
        spec=q_t
    ))
