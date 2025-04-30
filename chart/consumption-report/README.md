# Metrics Report Helm Chart

## How to install

1. Install Helm:

    ```bash
    $ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    $ chmod 700 get_helm.sh
    $ ./get_helm.sh
    ```

2. Add the metrics repo and perform a Helm upgrade.

    ```bash
    helm repo add metrics https://runai-professional-services.github.io/helm-charts/ && helm repo update
    ```

3. Export the values file and add your application credentials.

    ```bash
    helm show values metrics/consumption-report > metrics.yaml
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

7. Install the Helm chart with the following command:

    ```bash
    helm upgrade -i metrics -n runai-metrics metrics/consumption-report -f metrics.yaml
    ```

8. You should now have the following installed a Cronjob in the `runai-metrics` namespace called `metrics-consumption-report`. You can manually run the cronjob with the following command:

      ```bash
      kubectl -n runai-metrics create job metrics-job-01 --from=cronjob/metrics-consumption-report
      ```

9. You can get the logs from the job to confirm it was successful. 

    ```bash
    POD=$(kubectl -n runai-metrics get pods | grep -i metrics-job | awk '{print $1}')
    kubectl -n runai-metrics logs $POD --all-containers
    ```

10. During the install NOTES are provided on additional functionality. You can always view the notes by running:

    ```bash
    helm upgrade -i metrics -n runai-metrics metrics/consumption-report -f metrics.yaml --dry-run
    ```

## How to copy the .csv files locally

1. Download the v2 Run:ai cli.

    `https://run-ai-docs.nvidia.com/guides/reference/cli/install-cli`

2. Get the pvc name.

    ```bash
    PVC=$(kubectl -n runai-metrics get pvc | grep metrics | awk '{print $1}')
    ```

2. Submit a new workspace using the Run:ai cli.

    ```bash
    runai login
    runai workspace submit metrics -p metrics -i jupyter/scipy-notebook --existing-pvc claimname=$PVC,path=/mnt/data
    ```

3. Copy the `.csv` files to a local folder called `csv`.

    ```bash
    POD=$(kubectl -n runai-metrics get pods --sort-by=.metadata.creationTimestamp | tail -n 1 | awk '{print $1}')
    kubectl -n runai-metrics cp $POD:/mnt/data/ ./csv
    ```

4. Delete the workspace to free up access to the PVC.

    ```bash
    runai workload delete metrics -p metrics
    ```

## How to uninstall the Metrics Helm chart

1. You can delete the helm chart if needed by running the following:

    ```bash
    helm delete metrics -n metrics
    ```
