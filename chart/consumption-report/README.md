#### Metrics Report Helm Chart

### How to install

1. Install Helm:
```
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh
```

2. Add the metrics repo and perform a Helm upgrade.
`helm repo add metrics https://runai-professional-services.github.io/helm-charts/ && helm repo update`

3. Export the values file and add your application credentials
`helm show values metrics/consumption-report > metrics.yaml`

5. Create an application. This is used to provide credentials for API access. Here is the
documentation on how to create an `application`
`https://docs.run.ai/v2.19/admin/authentication/applications/?h=applications`

6. Make sure to provide your `application` with the appropriate role to provide access to
those resources when making an API call.

7. Modify the `prod-values.yaml`, make sure to update the following fields: The clientID and
clientSecret are the credentials from the `application` you created in step 5.
```
# Run:ai application credentials. Please provision a application to provide api access for the script.
credentials:
  clientId: "<name-of-application>"
  clientSecret: "<seceret-generated-on-creation>"
  baseUrl: "<base-url-to-runai>"

cron:
  # Define how often the metrics job is ran. Default is every midnight.
  schedule: "0 0 * * *"
```

8. Install the Helm chart with the following command:
`helm upgrade -i metrics -n runai . -f ./metrics.yaml`

9. You should now have the following installed:
    a. cronjob in the `runai` namespace called `metrics-consumption-report`. You can manually
    run the cronjob with the following command `kubectl create job <job-name> --from=cronjob/metrics-consumption-report`

10. During the install NOTES are provided on additional functionality. You can always view the
notes by running:
`helm upgrade -i metrics -n runai . -f ./metrics.yaml --dry-run`

### How to uninstall the Helm chart
1. You can delete the helm chart if needed by running the following
`helm delete metrics -n runai`
