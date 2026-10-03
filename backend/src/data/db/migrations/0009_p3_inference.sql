CREATE TABLE `p3_annotation_queue` (
	`id` bigint unsigned AUTO_INCREMENT NOT NULL,
	`inference_id` bigint unsigned NOT NULL,
	`image_id` bigint unsigned NOT NULL,
	`annotation_id` bigint unsigned,
	`status` enum('pending') NOT NULL DEFAULT 'pending',
	`created_at` timestamp(3) NOT NULL,
	CONSTRAINT `p3_annotation_queue_id` PRIMARY KEY(`id`),
	CONSTRAINT `p3_annotation_queue_inference_unique` UNIQUE(`inference_id`)
);
--> statement-breakpoint
CREATE TABLE `p3_inference` (
	`id` bigint unsigned AUTO_INCREMENT NOT NULL,
	`input_kind` enum('upload','crop') NOT NULL,
	`input` json NOT NULL,
	`input_sha256` char(64) NOT NULL,
	`storage_key` varchar(512),
	`predicted_class` varchar(16) NOT NULL,
	`probabilities` json NOT NULL,
	`model_source` enum('smoke','official') NOT NULL,
	`package_id` varchar(128) NOT NULL,
	`format_version` varchar(32) NOT NULL,
	`model_version` varchar(32),
	`mlflow_run_id` char(32) NOT NULL,
	`checkpoint_sha256` char(64) NOT NULL,
	`created_at` timestamp(3) NOT NULL,
	CONSTRAINT `p3_inference_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `p3_annotation_queue` ADD CONSTRAINT `p3_annotation_queue_inference_id_p3_inference_id_fk` FOREIGN KEY (`inference_id`) REFERENCES `p3_inference`(`id`) ON DELETE restrict ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `p3_annotation_queue` ADD CONSTRAINT `p3_annotation_queue_image_id_images_id_fk` FOREIGN KEY (`image_id`) REFERENCES `images`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `p3_inference_created_at_idx` ON `p3_inference` (`created_at`);