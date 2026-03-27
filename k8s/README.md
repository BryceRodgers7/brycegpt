## Kubernetes Deployment

This project uses Kubernetes to run and scale the GPT API. The
configuration is structured using Kustomize to separate
environment-agnostic resources from environment-specific overrides.

### Structure

    k8s/
      base/
        deployment.yaml
        service.yaml
      overlays/
        local/
        gke/

-   **base/** Contains shared Kubernetes resources that are
    environment-independent:

    -   Deployment
    -   Service
    -   Health probes
    -   Core environment variables

-   **overlays/local/** Local development configuration for Minikube:

    -   Uses locally built Docker image (`brycegpt-api:local`)
    -   Sets `imagePullPolicy: Never`
    -   No GCP credentials required (model is bundled with the
        container)

-   **overlays/gke/** Production configuration for Google Kubernetes
    Engine (GKE):

    -   Uses Artifact Registry image
    -   Enables Horizontal Pod Autoscaling (HPA)
    -   No Workload Identity required
    -   Exposes the service externally using GKE Ingress

------------------------------------------------------------------------

## Local Development (Minikube)

### Prerequisites

-   Docker Desktop
-   kubectl
-   Minikube
-   GCP SDK (`gcloud`) authenticated locally

------------------------------------------------------------------------

### Build and Load Image into Minikube

    docker build -t brycegpt-api:local .
    minikube image load brycegpt-api:local

------------------------------------------------------------------------

### Deploy to Kubernetes

    kubectl apply -k k8s/overlays/local

------------------------------------------------------------------------

### Test the Service

    kubectl port-forward service/gpt-api 8080:8080

Then in another terminal:

    curl http://localhost:8080/health

    curl -X POST "http://localhost:8080/generate" -H "Content-Type: application/json" --data-binary "@request.json"

------------------------------------------------------------------------

## Google Kubernetes Engine (GKE)

### Infrastructure (Terraform)

GKE infrastructure is provisioned using Terraform:

-   GKE cluster
-   Dedicated node pool with configurable machine type and size
-   Node service account for Artifact Registry access
-   Artifact Registry IAM binding (`roles/artifactregistry.reader`)

------------------------------------------------------------------------

### Build and Push Image

Before deploying to GKE, the container image must be built and pushed to
Artifact Registry:

    docker build -t us-central1-docker.pkg.dev/stoked-monitor-375412/ml-apps/bryce-gpt-api:prod .
    docker push us-central1-docker.pkg.dev/stoked-monitor-375412/ml-apps/bryce-gpt-api:prod

------------------------------------------------------------------------

### Deploy to GKE

    kubectl apply -k k8s/overlays/gke

------------------------------------------------------------------------

### Verify Deployment

    kubectl get pods
    kubectl get deployment
    kubectl get hpa
    kubectl get ingress

------------------------------------------------------------------------

### External Access (Ingress)

The GKE deployment uses a Kubernetes Ingress resource to expose the
service publicly.

-   Ingress is defined in `k8s/overlays/gke/ingress.yaml`
-   GKE automatically provisions an external HTTP load balancer
-   A public IP address is assigned to the Ingress

To retrieve the external IP:

    kubectl get ingress

Example output:

    NAME               ADDRESS
    gpt-api-ingress    34.x.x.x

------------------------------------------------------------------------

### Test via Public Endpoint

    curl http://<EXTERNAL_IP>/health

    curl -X POST "http://<EXTERNAL_IP>/generate" -H "Content-Type: application/json" --data-binary "@request.json"

------------------------------------------------------------------------

### Test via Port Forwarding (Optional)

    kubectl port-forward deployment/gpt-api 8080:8080

Then:

    curl http://localhost:8080/health

------------------------------------------------------------------------

## Model Handling

The GPT model is bundled directly within the container image.

-   No external storage (e.g., GCS) is required
-   No credential files or Workload Identity are needed
-   Model is loaded directly from the application code at startup

------------------------------------------------------------------------

## Scaling

The GPT API supports horizontal scaling:

-   Local: manually set replica count in overlay
-   GKE: uses Horizontal Pod Autoscaler (HPA)

Example:

    kubectl get pods

Multiple pods will be created and managed automatically.

------------------------------------------------------------------------

## Architecture Summary

    Frontend
       │
       ├── Cloud Run (current production)
       │
       └── GKE (Kubernetes deployment)
             │
             ├── Ingress (external load balancer)
             │
             ├── Service (internal networking)
             │
             ├── Deployment (gpt-api)
             │     └── Pods (replicated)
             │
             ├── HPA (auto-scaling)
             │
             └── Container Image (Artifact Registry)

------------------------------------------------------------------------

## Deployment Tradeoffs

This project includes both Cloud Run and GKE deployments of the same
API.

### Cloud Run

-   Fully managed, serverless
-   Scales to zero
-   Lower operational overhead
-   Cost-efficient for low or bursty traffic

### GKE (Standard)

-   Full Kubernetes control
-   Supports complex workloads and scaling patterns
-   Requires always-on compute (nodes)
-   Higher operational overhead and cost

For this project, Cloud Run is used as the primary serving layer, while
GKE is implemented for learning, flexibility, and demonstrating
Kubernetes-based deployment patterns.
