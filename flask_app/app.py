from flask import Flask, render_template, jsonify, request
from CollisionVision.database.database import DatabaseManager
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(APP_ROOT, 'templates'),
    static_folder=os.path.join(APP_ROOT, 'static')
)

db = DatabaseManager()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/collision_events')
def get_collision_events():
    # get the latest events from this table
    events = db.get_recent_events(table_name="collision_events")
    return jsonify(events)

@app.route('/api/system_metrics')
def get_system_metrics():
    metrics = db.get_recent_events(table_name="system_metrics")
    return jsonify(metrics)


if __name__ == '__main__':
    app.run()
