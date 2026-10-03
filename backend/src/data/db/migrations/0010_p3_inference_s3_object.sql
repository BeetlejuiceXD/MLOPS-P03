ALTER TABLE `p3_inference` ADD `s3_bucket` varchar(63);--> statement-breakpoint
ALTER TABLE `p3_inference` ADD `s3_key` varchar(1024);--> statement-breakpoint
ALTER TABLE `p3_inference` ADD `s3_version_id` varchar(1024);--> statement-breakpoint
ALTER TABLE `p3_inference` ADD `s3_sha256` char(64);