/**
 * SQLite Database Connection and Schema Management.
 */

const fs = require('fs');
const path = require('path');
const Database = require('better-sqlite3');

const DB_DIR = path.resolve(__dirname, '../../data');
if (!fs.existsSync(DB_DIR)) {
  fs.mkdirSync(DB_DIR, { recursive: true });
}

const DB_PATH = process.env.DATABASE_PATH || path.join(DB_DIR, 'acoustic_monitor.sqlite');
const db = new Database(DB_PATH);

// Enable foreign keys and WAL mode for high concurrency
db.pragma('foreign_keys = ON');
db.pragma('journal_mode = WAL');

// Initialize tables as specified in README Section 21
db.exec(`
  CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_type TEXT NOT NULL DEFAULT 'unknown',
    audio_path TEXT NOT NULL,
    timestamp REAL DEFAULT 0.0,
    duration REAL DEFAULT 0.0,
    sample_rate INTEGER DEFAULT 16000,
    gate_type TEXT NOT NULL,
    triggered INTEGER NOT NULL DEFAULT 0,
    trigger_score REAL DEFAULT 0.0,
    gate_metadata TEXT,
    processing_status TEXT NOT NULL DEFAULT 'PENDING',
    spectrogram_path TEXT,
    created_at TEXT DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    result TEXT NOT NULL,
    anomaly_score REAL,
    confidence REAL,
    model_version TEXT,
    processing_time REAL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    gate_type TEXT NOT NULL,
    dataset TEXT NOT NULL,
    config TEXT,
    metrics TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
  );

  CREATE INDEX IF NOT EXISTS idx_events_triggered ON events(triggered);
  CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
  CREATE INDEX IF NOT EXISTS idx_predictions_event_id ON predictions(event_id);
`);

module.exports = db;
