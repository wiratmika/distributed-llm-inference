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
  count        = 3
  name         = "gateway-${count.index + 1}"
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

  metadata_startup_script = file("${path.module}/scripts/setup-vm.sh")
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

  metadata_startup_script = file("${path.module}/scripts/setup-vm.sh")
}
