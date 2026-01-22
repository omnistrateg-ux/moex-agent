PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS candles (
  secid TEXT NOT NULL,
  board TEXT NOT NULL,
  interval INTEGER NOT NULL,
  ts TEXT NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  value REAL,
  volume REAL,
  PRIMARY KEY (secid, board, interval, ts)
);

CREATE TABLE IF NOT EXISTS quotes (
  secid TEXT NOT NULL,
  board TEXT NOT NULL,
  ts TEXT NOT NULL,
  last REAL,
  bid REAL,
  ask REAL,
  numtrades REAL,
  voltoday REAL,
  valtoday REAL,
  PRIMARY KEY (secid, board, ts)
);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_ts TEXT NOT NULL,
  secid TEXT NOT NULL,
  horizon TEXT NOT NULL,
  p REAL NOT NULL,
  signal_type TEXT NOT NULL,
  entry REAL,
  take REAL,
  stop REAL,
  ttl_minutes INTEGER,
  anomaly_score REAL,
  payload_json TEXT,
  sent INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS state (
  key TEXT PRIMARY KEY,
  value TEXT
);
