CREATE TABLE `training_job_logs` (
	`id` bigint unsigned AUTO_INCREMENT NOT NULL,
	`job_id` bigint unsigned NOT NULL,
	`ts` timestamp(3) NOT NULL DEFAULT (now()),
	`level` enum('info','warning','error') NOT NULL,
	`message` text NOT NULL,
	CONSTRAINT `training_job_logs_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `training_jobs` (
	`id` bigint unsigned AUTO_INCREMENT NOT NULL,
	`task` enum('controlled','training') NOT NULL,
	`status` enum('queued','running','succeeded','failed','cancelled') NOT NULL DEFAULT 'queued',
	`dataset_version` varchar(32) NOT NULL,
	`manifest_hash` char(64) NOT NULL,
	`config` json NOT NULL,
	`controlled_fail_at_epoch` int unsigned,
	`progress_epoch` int unsigned,
	`total_epochs` int unsigned,
	`mlflow_run_id` char(32),
	`error` text,
	`cancel_requested` boolean NOT NULL DEFAULT false,
	`worker_id` varchar(128),
	`heartbeat_at` timestamp NULL DEFAULT NULL,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	`started_at` timestamp NULL DEFAULT NULL,
	`finished_at` timestamp NULL DEFAULT NULL,
	`updated_at` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `training_jobs_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `training_job_logs` ADD CONSTRAINT `training_job_logs_job_id_training_jobs_id_fk` FOREIGN KEY (`job_id`) REFERENCES `training_jobs`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `training_job_logs_job_idx` ON `training_job_logs` (`job_id`,`id`);--> statement-breakpoint
CREATE INDEX `training_jobs_status_created_idx` ON `training_jobs` (`status`,`created_at`);