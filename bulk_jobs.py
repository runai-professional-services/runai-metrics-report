import random
import string
from runai.client import RunaiClient


def random_string(length: int):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length)).lower()


client = RunaiClient(
    client_id="api-test",
    client_secret="2I84hkD7HLxL1cPYD5nkBRl6p1EEmCUA",
    runai_base_url="https://envinaclick.run.ai",
    cluster_id="461619fd-127a-4cc6-979c-5cd843a37a2d",
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
        project_id="4513786",
        cluster_id="461619fd-127a-4cc6-979c-5cd843a37a2d",
        spec=q_t
    ))