output "bucket_name" {
  description = "Nombre REAL del bucket de modelos (asignado al crearlo): el valor de MODEL_S3_BUCKET."
  value       = aws_s3_bucket.models.bucket
}

output "bucket_arn" {
  description = "ARN del bucket de modelos."
  value       = aws_s3_bucket.models.arn
}

output "model_key_prefix" {
  description = "Prefijo de las claves del registro: models/p3-cnn-classifier/<semver>/."
  value       = local.model_key_prefix
}

output "operational_policy_json" {
  description = "Política de mínimo privilegio para el permission set MLOPS-S3-MODELS."
  value       = local.operational_policy
}
