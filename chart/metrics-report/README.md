# Metrics Report Helm Chart

## How to install
This section will guide you through the process of deploying the metrics report as a Helm chart. This will deploy a `PVC` to store the reports and a `Cron` job which will schedule when the report is ran. Additonally there are steps provided to access the reports via a Run:AI workspace.

1. Install Helm.

    ```bash
    $ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    $ chmod 700 get_helm.sh
    $ ./get_helm.sh
    ```

2. Add the metrics repo and perform a Helm update.

    ```bash
    helm repo add metrics https://runai-professional-services.github.io/runai-metrics-report && helm repo update
    ```

3. Export the values file and add your application credentials.

    ```bash
    helm show values metrics/metrics-report > metrics.yaml
    ```

4. Create an application. This is used to provide credentials for API access. Here is the documentation on how to create an `application`.

    `https://docs.run.ai/v2.19/admin/authentication/applications/?h=applications`

5. Make sure to provide your `application` with the appropriate role to provide access to resources when making an API call.

6. Modify the `metrics.yaml`, make sure to update the following fields: The `clientID` and `clientSecret` are the credentials from the `application` you created in step 5.

    ```bash
    # Run:ai application credentials. Please provision a application to provide api access for the script.
    credentials:
      clientId: "<name-of-application>"
      clientSecret: "<seceret-generated-from-application>"
      baseUrl: "<https://base-url-to-runai>"

    cron:
      # Define how often the metrics job is ran. Default is every midnight.
      schedule: "0 0 * * *"
    ```

7. Create a new Project in the Run:AI UI called `metrics`. We will use this project namespace to manage easy access to the report files located on the PVC created by the `helm install`.

8. Install the Helm chart with the following command:

    ```bash
    helm upgrade -i metrics -n runai-metrics metrics/metrics-report -f metrics.yaml
    ```

9. You should now have the following installed a `Cronjob` in the `runai-metrics` namespace called `metrics-metrics-report`. You can manually run the cronjob with the following command:

      ```bash
      kubectl -n runai-metrics create job metrics-job-01 --from=cronjob/metrics-metrics-report
      ```

10. You can get the logs from the job to confirm it was successful. 

    ```bash
    POD=$(kubectl -n runai-metrics get pods | grep -i metrics-job | awk '{print $1}')
    kubectl -n runai-metrics logs $POD --all-containers
    ```

11. During the install NOTES are provided on additional functionality. You can always view the notes by running:

    ```bash
    helm upgrade -i metrics -n runai-metrics metrics/metrics-report -f metrics.yaml --dry-run
    ```

## How to copy the .csv files locally

1. Download the v2 Run:AI cli.

    `https://run-ai-docs.nvidia.com/guides/reference/cli/install-cli`

2. Get the `metrics` pvc name.

    ```bash
    PVC=$(kubectl -n runai-metrics get pvc | grep metrics | awk '{print $1}')
    ```

3. Submit a new workspace using the Run:AI cli.

    ```bash
    runai login
    runai workspace submit metrics -p metrics -i jupyter/scipy-notebook --existing-pvc claimname=$PVC,path=/mnt/data
    ```

4. Copy the `.csv` files to a local folder called `csv`.

    ```bash
    POD=$(kubectl -n runai-metrics get pods --sort-by=.metadata.creationTimestamp | tail -n 1 | awk '{print $1}')
    kubectl -n runai-metrics cp $POD:/mnt/data/ ./csv
    ```

5. Delete the workspace to free up access to the PVC.

    ```bash
    runai workload delete metrics -p metrics
    ```

## How to Upgrade the Metrics Helm chart

1. Save your current values for the metrics helm chart:

    ```bash
    helm get values metrics -n runai-metrics > metrics.yaml
    ```

2. Perform a Helm update to get the latest chart:

    ```bash
    helm repo update
    ```

3. Perorm a Helm upgrade to update the Metrics Helm release. If you don't specify the `--version` the latest will be used. 

    **Optionally** you can specify the exact chart version as shown below.

    ```bash
    helm upgrade metrics -n runai-metrics metrics/metrics-report \
    -f metrics.yaml \
    --version=0.0.4 # If not specified the latest vesions will be used
    ```

## How to Uninstall the Metrics Helm chart

1. You can delete the helm chart if needed by running the following:

    ```bash
    helm delete metrics -n runai-metrics
    ```
