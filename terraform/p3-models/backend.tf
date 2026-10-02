terraform {
  # Mismo bucket de estado remoto que P2-25; bucket y región con terraform init -backend-config.
  backend "s3" {
    key          = "p3-models/terraform.tfstate"
    use_lockfile = true
    encrypt      = true
  }
}
