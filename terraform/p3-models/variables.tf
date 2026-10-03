variable "region" {
  description = "Región del bucket de modelos (la del resto del proyecto)."
  type        = string
  default     = "us-east-1"
}

variable "bucket_prefix" {
  description = "Prefijo del bucket exclusivo de modelos P3. El nombre real lo asigna AWS al crearlo y se lee del output bucket_name."
  type        = string
  default     = "mlops-p3-models-"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{2,35}-$", var.bucket_prefix))
    error_message = "bucket_prefix: minúsculas, dígitos y guiones, terminado en guion (máximo 37 caracteres)."
  }
}

variable "create_operational_policy" {
  description = "Crear la política IAM del permiso operacional. Por defecto no: el JSON se entrega para el permission set MLOPS-S3-MODELS de IAM Identity Center."
  type        = bool
  default     = false
}
