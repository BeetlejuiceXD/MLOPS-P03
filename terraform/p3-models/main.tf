# D06-03 — Bucket EXCLUSIVO de modelos P3 (#33). No adopta ni referencia los buckets de
# DVC ni de artefactos. El registro de D04-06 escribe y lee cada versión del modelo por
# su VersionId exacto en models/p3-cnn-classifier/<semver>/.

locals {
  model_key_prefix = "models/p3-cnn-classifier/"
}

resource "aws_s3_bucket" "models" {
  bucket_prefix = var.bucket_prefix
  force_destroy = false

  tags = { Name = "mlops-p3-models" }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Sin lifecycle: las versiones anteriores se conservan (P3-14). La política niega el acceso
# sin TLS y borrar versiones o el bucket, también a quien tenga s3:* por otra vía.
resource "aws_s3_bucket_policy" "models" {
  bucket = aws_s3_bucket.models.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      },
      {
        Sid       = "DenyDeleteModelVersions"
        Effect    = "Deny"
        Principal = "*"
        Action    = ["s3:DeleteObjectVersion", "s3:DeleteBucket"]
        Resource  = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
      },
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.models]
}
