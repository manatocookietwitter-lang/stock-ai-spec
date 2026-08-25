CREATE TABLE `operator_settings` (
	`user_id` text PRIMARY KEY NOT NULL,
	`safety_acknowledged` integer DEFAULT false NOT NULL,
	`updated_at` text NOT NULL
);
