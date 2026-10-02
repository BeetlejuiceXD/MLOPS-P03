# D06-03 — Permiso OPERACIONAL de mínimo privilegio para publicar y recuperar el modelo
# (MLOPS-S3-MODELS). Es distinto de la identidad que aplica este Terraform
# (MLOPS-P3-Terraform) y de MLOPS-S3-DVC. Sin borrar nada; solo bajo models/p3-cnn-classifier/.

locals {
  operational_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "PublishAndReadModelVersions"
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:GetObjectVersion"]
        Resource = "${aws_s3_bucket.models.arn}/${local.model_key_prefix}*"
      },
      {
        Sid       = "ListModelPrefix"
        Effect    = "Allow"
        Action    = ["s3:ListBucket", "s3:ListBucketVersions"]
        Resource  = aws_s3_bucket.models.arn
        Condition = { StringLike = { "s3:prefix" = ["${local.model_key_prefix}*"] } }
      },
      {
        Sid      = "ReadBucketSecurity"
        Effect   = "Allow"
        Action   = ["s3:GetBucketVersioning", "s3:GetEncryptionConfiguration", "s3:GetBucketPublicAccessBlock", "s3:GetBucketPolicy", "s3:GetLifecycleConfiguration"]
        Resource = aws_s3_bucket.models.arn
      },
    ]
  })
}

resource "aws_iam_policy" "operational" {
  count       = var.create_operational_policy ? 1 : 0
  name        = "mlops-p3-s3-models-operational"
  description = "D06-03: publicar y leer por VersionId el modelo P3 (sin borrar)."
  policy      = local.operational_policy
}
