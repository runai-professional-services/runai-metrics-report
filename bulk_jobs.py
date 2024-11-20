from runai.client import RunaiClient

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
for i in range(20):
    client.training.create(
        training_name="quickstart",
        use_given_name_as_prefix=True,
        project_id="4513786",
        cluster_id="461619fd-127a-4cc6-979c-5cd843a37a2d",
        spec=q_t
    )