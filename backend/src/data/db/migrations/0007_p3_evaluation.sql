CREATE TABLE `p3_evaluation` (
	`namespace` enum('official','synthetic') NOT NULL,
	`candidate_run_id` char(32) NOT NULL,
	`evaluated_at` timestamp(3) NOT NULL,
	`evaluation` json NOT NULL,
	`predictions` json NOT NULL,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `p3_evaluation_namespace` PRIMARY KEY(`namespace`)
);
