CREATE TABLE `p3_training_sources` (
	`name` enum('releases','manifest') NOT NULL,
	`status` enum('ok','unavailable') NOT NULL,
	`payload` text,
	`detail` text,
	`updated_at` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `p3_training_sources_name` PRIMARY KEY(`name`)
);
