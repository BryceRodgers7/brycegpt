terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.25"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "container_api" {
  project = var.project_id
  service = "container.googleapis.com"
}

resource "google_container_cluster" "gpt_cluster" {
  depends_on = [google_project_service.container_api]

  name     = var.cluster_name
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  networking_mode     = "VPC_NATIVE"
  deletion_protection = false

  release_channel {
    channel = "REGULAR"
  }
}

resource "google_container_node_pool" "primary_nodes" {
  name       = var.node_pool_name
  location   = var.region
  cluster    = google_container_cluster.gpt_cluster.name
  node_count = var.node_count

  node_config {
    machine_type = var.machine_type
    disk_size_gb = var.disk_size_gb
    disk_type    = "pd-balanced"


    service_account = google_service_account.gke_node_sa.email

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      workload = "gpt-api"
      env      = var.environment
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_service_account" "gke_node_sa" {
  account_id   = "gke-node-sa"
  display_name = "GKE Node Service Account"
}

resource "google_artifact_registry_repository_iam_member" "node_sa_reader" {
  location   = var.region
  repository = var.artifact_registry_repo
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.gke_node_sa.email}"
}
