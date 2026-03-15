output "gateway_external_ips" {
  value = [
    for instance in google_compute_instance.gateways :
    instance.network_interface[0].access_config[0].nat_ip
  ]
  description = "Gateway public IP addresses"
}

output "gateway_internal_ips" {
  value = [
    for instance in google_compute_instance.gateways :
    instance.network_interface[0].network_ip
  ]
  description = "Gateway internal IP addresses"
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
    gateways = [
      "gcloud compute ssh gateway --zone=${var.zone}"
    ]
    workers = [
      for i in range(7) :
      "gcloud compute ssh worker-${i + 1} --zone=${var.zone}"
    ]
  }
}
