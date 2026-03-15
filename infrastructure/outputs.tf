output "gateway_external_ip" {
  value       = google_compute_instance.gateway.network_interface[0].access_config[0].nat_ip
  description = "Gateway public IP address"
}

output "gateway_internal_ip" {
  value       = google_compute_instance.gateway.network_interface[0].network_ip
  description = "Gateway internal IP address"
}

output "worker_external_ips" {
  value = [
    for instance in google_compute_instance.workers :
    instance.network_interface[0].access_config[0].nat_ip
  ]
  description = "Worker public IP addresses"
}

output "worker_internal_ips" {
  value = [
    for instance in google_compute_instance.workers :
    instance.network_interface[0].network_ip
  ]
  description = "Worker internal IP addresses"
}

output "ssh_commands" {
  value = {
    gateway = "gcloud compute ssh gateway --zone=${var.zone}"
    workers = [
      for i in range(7) :
      "gcloud compute ssh worker-${i + 1} --zone=${var.zone}"
    ]
  }
}
