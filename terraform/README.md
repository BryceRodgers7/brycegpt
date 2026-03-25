# Cloud Run Terraform - brycegpt

This Terraform manages the Google Cloud Run service configuration for `brycegpt`.

Managed here:
- service name
- region
- ingress
- public access
- scaling limits
- CPU / memory
- timeout
- startup CPU boost
- service account reference

Not managed here yet:
- Cloud Build trigger
- service account creation
- Artifact Registry IAM
- dynamic build labels / revision metadata

Notes:
- Cloud Build continues to deploy new image revisions on push to `main`.
- Some Cloud Run system/build metadata is ignored via `lifecycle.ignore_changes`.