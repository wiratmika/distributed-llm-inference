terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

locals {
  internal_dns_suffix = "${var.zone}.c.${var.project_id}.internal"

  worker_layout = [
    { num_nodes = 1, rank = 0, next_node_url = "", port = 8001 },
    { num_nodes = 2, rank = 0, next_node_url = "http://worker-3.${local.internal_dns_suffix}:8003", port = 8002 },
    { num_nodes = 2, rank = 1, next_node_url = "", port = 8003 },
    { num_nodes = 4, rank = 0, next_node_url = "http://worker-5.${local.internal_dns_suffix}:8005", port = 8004 },
    { num_nodes = 4, rank = 1, next_node_url = "http://worker-6.${local.internal_dns_suffix}:8006", port = 8005 },
    { num_nodes = 4, rank = 2, next_node_url = "http://worker-7.${local.internal_dns_suffix}:8007", port = 8006 },
    { num_nodes = 4, rank = 3, next_node_url = "", port = 8007 },
  ]
}

# VPC Network
resource "google_compute_network" "llm_network" {
  name                    = "llm-inference-network"
  auto_create_subnetworks = true
}

# Firewall rule for HTTP traffic
resource "google_compute_firewall" "llm_firewall" {
  name    = "llm-inference-allow"
  network = google_compute_network.llm_network.name

  allow {
    protocol = "tcp"
    ports    = ["8000-8007", "22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["llm-inference"]
}

# Gateway VMs
resource "google_compute_instance" "gateways" {
  count        = 1
  name         = "gateway"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["llm-inference"]

  scheduling {
    provisioning_model  = "SPOT"
    preemptible         = true
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.boot_disk_size_gb
    }
  }

  network_interface {
    network = google_compute_network.llm_network.name
    access_config {
      # Ephemeral public IP
    }
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(pathexpand(var.ssh_public_key_path))}"
  }

  metadata_startup_script = templatefile("${path.module}/scripts/startup-gateway.sh.tpl", {
    model_name   = var.model_name
    app_port     = 8000
    worker_url_1 = "http://worker-1.${local.internal_dns_suffix}:8001"
    worker_url_2 = "http://worker-2.${local.internal_dns_suffix}:8002"
    worker_url_4 = "http://worker-4.${local.internal_dns_suffix}:8004"
  })
}

# Worker VMs
resource "google_compute_instance" "workers" {
  count        = 7
  name         = "worker-${count.index + 1}"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["llm-inference"]

  scheduling {
    provisioning_model  = "SPOT"
    preemptible         = true
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.boot_disk_size_gb
    }
  }

  network_interface {
    network = google_compute_network.llm_network.name
    access_config {
      # Ephemeral public IP
    }
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(pathexpand(var.ssh_public_key_path))}"
  }

  metadata_startup_script = templatefile("${path.module}/scripts/startup-worker.sh.tpl", {
    worker_id     = count.index + 1
    model_name    = var.model_name
    num_nodes     = local.worker_layout[count.index].num_nodes
    rank          = local.worker_layout[count.index].rank
    app_port      = local.worker_layout[count.index].port
    next_node_url = local.worker_layout[count.index].next_node_url
  })
}
