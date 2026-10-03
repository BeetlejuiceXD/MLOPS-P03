CREATE TABLE `p3_model_card` (
	`namespace` enum('official','local_test') NOT NULL,
	`semver` varchar(32) NOT NULL,
	`mlflow_run_id` char(32) NOT NULL,
	`manifest_hash` char(64) NOT NULL,
	`dvc_release` varchar(32) NOT NULL,
	`dvc_release_hash` char(64) NOT NULL,
	`s3_bucket` varchar(63) NOT NULL,
	`s3_key` varchar(512) NOT NULL,
	`version_id` varchar(1024),
	`sha256` char(64) NOT NULL,
	`size_bytes` bigint unsigned NOT NULL,
	`status` enum('draft','published','failed') NOT NULL,
	`failure_reason` varchar(32),
	`failure_detail` text,
	`published_at` timestamp(3),
	`created_at` timestamp(3) NOT NULL DEFAULT (now()),
	CONSTRAINT `p3_model_card_namespace_semver_pk` PRIMARY KEY(`namespace`,`semver`)
);
