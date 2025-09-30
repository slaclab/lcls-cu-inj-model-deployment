# LCLS CU Injector ML Model Deployment
This is the repository for online deployment of the LCLS CU Injector ML Model. It's still under development.

It was created using the lume-model-deployment-template using `copier`.

## To check deployment
First set up kubectl by following the instructions in this [link](https://k8s.slac.stanford.edu/ad-accel-online-ml).
To see the deployment logs of the container, you can use the following commands to get the name of the pod 
that's running, and then view the logs:
```bash
kubectl get pods
kubectl logs -f <pod-name>
```
## Docker image deployment on Kubernetes (S3DF)
To deploy the Docker image on Kubernetes, follow these steps. If you have updated the tag,
make sure to replace `<tag>` with the new tag in the commands below, **and in the deployment YAML file**.

0. Ensure you have Docker installed and running on your machine.
1. Update the `deployment.yaml` file with the correct image tag and registry information, if that information has changed.
2. Build the Docker image. The `platform` tag is necessary if you are developing on a machine with a different architecture. 
If you are NOT building on a MacOS machine, you can skip the `--provenance` flag.
   ```bash
   docker build -t <image-name>:<tag> . --platform=linux/amd64 --provenance=false
   ```
2. Push the Docker image to the Stanford Container Registry (replace `<your-username>` with your actual username):
    ```bash
    cat ~/.scr-token | docker login --username $USER --password-stdin http://scr.svc.stanford.edu
    docker tag <image-name>:<tag> scr.svc.stanford.edu/<your-username>/<image-name>:<tag>
    docker push scr.svc.stanford.edu/<your-username>/<image-name>:<tag>
    ```
   
For more detailed instructions, refer to this [documentation](https://github.com/slaclab/lcls_cu_injector_ml_model?tab=readme-ov-file#containerization-steps).

3. To set up kubectl, follow this [link](https://k8s.slac.stanford.edu/lcls-ml-online). Update the Kubernetes deployment with the new image:
    ```bash
    kubectl apply -f deployment.yaml
    ```
   
## Accessing the deployed image
Once the image is deployed, you can access following these steps:
1. List all pods to find the name of the pod running the image:
   ```bash
   kubectl get pods
   ```
2. Use the following command to access the pod:
   ```bash
   kubectl exec -ti <pod-name> -- bash
    ```
3. Once inside the pod, you can run the Python scripts or other commands as needed. To run the 
tests, you can run for example (make sure you're in src/ directory):
   ```bash
   python -m <python_script>
   ```
    or
    
    ```bash
    pytest tests/test.py -s -rsx -v
    ```
    If you want to edit the code, you can use a text editor like `vim` to modify the files directly within the pod. 
    For example if you want to edit a file to write/not write to PV:
   ```bash
   apt update && apt install vim
   vi tests/test.py
   ```