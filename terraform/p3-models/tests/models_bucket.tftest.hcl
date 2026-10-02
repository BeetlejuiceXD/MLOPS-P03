# D06-03 — Bucket exclusivo de modelos P3 y permiso operacional de mínimo privilegio.
#
# Corre con `terraform test` y un provider SIMULADO: no necesita credenciales ni toca AWS.
# Comprueba la configuración que Terraform aplicaría; el bucket real se acredita aparte
# (plan/apply con MLOPS-P3-Terraform y lectura de su configuración en AWS).

mock_provider "aws" {}

override_resource {
  target = aws_s3_bucket.models
  values = {
    id     = "mlops-p3-models-test"
    bucket = "mlops-p3-models-test"
    arn    = "arn:aws:s3:::mlops-p3-models-test"
  }
}

run "bucket_exclusivo_y_seguro" {
  command = apply

  assert {
    condition     = aws_s3_bucket.models.bucket_prefix == "mlops-p3-models-"
    error_message = "El bucket de modelos es exclusivo de P3 (prefijo mlops-p3-models-), no el de DVC ni el de artefactos."
  }

  assert {
    condition     = aws_s3_bucket.models.force_destroy == false
    error_message = "force_destroy debe ser false: borrar el bucket nunca puede llevarse las versiones."
  }

  assert {
    condition     = aws_s3_bucket_versioning.models.versioning_configuration[0].status == "Enabled"
    error_message = "Versioning debe estar Enabled (el registro lee por VersionId exacto)."
  }

  assert {
    condition     = [for rule in aws_s3_bucket_server_side_encryption_configuration.models.rule : one(rule.apply_server_side_encryption_by_default).sse_algorithm] == ["AES256"]
    error_message = "Cifrado en reposo SSE AES256."
  }

  assert {
    condition = alltrue([
      aws_s3_bucket_public_access_block.models.block_public_acls,
      aws_s3_bucket_public_access_block.models.block_public_policy,
      aws_s3_bucket_public_access_block.models.ignore_public_acls,
      aws_s3_bucket_public_access_block.models.restrict_public_buckets,
    ])
    error_message = "El bloqueo de acceso público debe estar completo."
  }

  assert {
    condition     = [for rule in aws_s3_bucket_ownership_controls.models.rule : rule.object_ownership] == ["BucketOwnerEnforced"]
    error_message = "Sin ACL: BucketOwnerEnforced."
  }

  assert {
    condition = anytrue([
      for statement in jsondecode(aws_s3_bucket_policy.models.policy).Statement :
      try(statement.Sid == "DenyInsecureTransport" && statement.Effect == "Deny" && statement.Condition.Bool["aws:SecureTransport"] == "false", false)
    ])
    error_message = "La política del bucket debe negar el acceso sin TLS (DenyInsecureTransport)."
  }

  assert {
    condition = anytrue([
      for statement in jsondecode(aws_s3_bucket_policy.models.policy).Statement :
      try(statement.Sid == "DenyDeleteModelVersions" && statement.Effect == "Deny" && contains(statement.Action, "s3:DeleteObjectVersion"), false)
    ])
    error_message = "Las versiones del modelo se conservan: la política niega DeleteObjectVersion."
  }

  assert {
    condition     = output.model_key_prefix == "models/p3-cnn-classifier/"
    error_message = "Prefijo de las claves del registro: models/p3-cnn-classifier/<semver>/."
  }

  assert {
    condition     = output.bucket_name == "mlops-p3-models-test"
    error_message = "El nombre real del bucket sale del recurso creado, no de una variable inventada."
  }
}

run "permiso_operacional_de_minimo_privilegio" {
  command = apply

  assert {
    condition = alltrue(flatten([
      for statement in jsondecode(output.operational_policy_json).Statement : [
        for action in flatten([statement.Action]) : !strcontains(action, "Delete") && action != "s3:*"
      ]
    ]))
    error_message = "El permiso operacional no borra nada ni usa s3:*."
  }

  assert {
    condition = toset(flatten([
      for statement in jsondecode(output.operational_policy_json).Statement : statement.Action
      if statement.Sid == "PublishAndReadModelVersions"
    ])) == toset(["s3:PutObject", "s3:GetObject", "s3:GetObjectVersion"])
    error_message = "Objetos: solo subir y leer (incluida una versión concreta por VersionId)."
  }

  assert {
    condition = alltrue([
      for statement in jsondecode(output.operational_policy_json).Statement :
      statement.Resource == "arn:aws:s3:::mlops-p3-models-test/models/p3-cnn-classifier/*"
      if statement.Sid == "PublishAndReadModelVersions"
    ])
    error_message = "Los objetos se limitan al prefijo models/p3-cnn-classifier/ del bucket de modelos."
  }

  assert {
    condition = anytrue([
      for statement in jsondecode(output.operational_policy_json).Statement :
      try(statement.Sid == "ListModelPrefix" && statement.Condition.StringLike["s3:prefix"] == ["models/p3-cnn-classifier/*"], false)
    ])
    error_message = "Listar solo bajo el prefijo de modelos."
  }

  assert {
    condition = toset(flatten([
      for statement in jsondecode(output.operational_policy_json).Statement : statement.Action
      if statement.Sid == "ReadBucketSecurity"
    ])) == toset(["s3:GetBucketVersioning", "s3:GetEncryptionConfiguration", "s3:GetBucketPublicAccessBlock", "s3:GetBucketPolicy", "s3:GetLifecycleConfiguration"])
    error_message = "Antes de publicar se lee (sin cambiarla) la configuración de seguridad del bucket."
  }

  assert {
    condition     = length(aws_iam_policy.operational) == 0
    error_message = "Por defecto Terraform no crea IAM: la política se entrega para el permission set MLOPS-S3-MODELS."
  }
}

run "politica_iam_opcional" {
  command = apply

  variables {
    create_operational_policy = true
  }

  assert {
    condition     = length(aws_iam_policy.operational) == 1 && aws_iam_policy.operational[0].name == "mlops-p3-s3-models-operational"
    error_message = "Si se pide, la política operacional se crea con su nombre propio."
  }

  assert {
    condition     = aws_iam_policy.operational[0].policy == output.operational_policy_json
    error_message = "La política creada es la misma que se entrega para el permission set."
  }
}

run "prefijo_invalido" {
  command = plan

  variables {
    bucket_prefix = "MLOPS_Modelos"
  }

  expect_failures = [var.bucket_prefix]
}
