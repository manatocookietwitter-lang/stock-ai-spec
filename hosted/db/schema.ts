import { integer, sqliteTable, text } from 'drizzle-orm/sqlite-core';

export const operatorSettings = sqliteTable('operator_settings', {
  userId: text('user_id').primaryKey(),
  safetyAcknowledged: integer('safety_acknowledged', { mode: 'boolean' })
    .notNull()
    .default(false),
  updatedAt: text('updated_at').notNull(),
});
