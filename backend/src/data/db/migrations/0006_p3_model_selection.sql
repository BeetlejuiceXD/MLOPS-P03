CREATE TABLE `p3_model_selection` (
	`id` int NOT NULL,
	`status` enum('open','candidate','closed') NOT NULL,
	`outcome` json,
	`outcome_hash` char(64),
	`proposed_at` timestamp(3) NULL DEFAULT NULL,
	`closed_at` timestamp(3) NULL DEFAULT NULL,
	CONSTRAINT `p3_model_selection_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
INSERT INTO `p3_model_selection` (`id`, `status`) VALUES (1, 'open');
