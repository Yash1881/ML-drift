import os
import sqlite3
import json
import datetime
import numpy as np

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drift_logs.db')

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS drift_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT,
            model_health TEXT,
            prediction_drift_psi REAL,
            feature_results_json TEXT,
            performance_metrics_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_drift_run(filename, model_health, prediction_drift_psi, feature_results, performance_metrics=None):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    feature_results_json = json.dumps(feature_results, cls=NumpyEncoder)
    performance_metrics_json = json.dumps(performance_metrics, cls=NumpyEncoder) if performance_metrics else "{}"
    
    c.execute('''
        INSERT INTO drift_runs (timestamp, filename, model_health, prediction_drift_psi, feature_results_json, performance_metrics_json)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, filename, model_health, prediction_drift_psi, feature_results_json, performance_metrics_json))
    
    conn.commit()
    conn.close()

def get_drift_history():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT * FROM drift_runs ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'filename': row['filename'],
            'model_health': row['model_health'],
            'prediction_drift_psi': row['prediction_drift_psi'],
            'feature_results': json.loads(row['feature_results_json']),
            'performance_metrics': json.loads(row['performance_metrics_json'])
        })
    return results
