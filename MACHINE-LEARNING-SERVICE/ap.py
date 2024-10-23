from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS, cross_origin
from sqlalchemy import create_engine
import os
import logging

from utils.spatial_operations import run_full_spatial_operations

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize Flask-CORS before defining any routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize SocketIO with threading async mode
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    logger=True, 
    engineio_logger=True, 
    async_mode='threading'  # Changed from 'eventlet' to 'threading'
)

# Database connection setup
connection_params = {
    'dbname': 'GEGIS2',
    'user': 'postgres',
    'password': 'p#maki012412',
    'host': 'localhost',
    'port': '5432'
}

engine = create_engine(
    f"postgresql://{connection_params['user']}:{connection_params['password']}@"
    f"{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}"
)

@app.route('/start_spatial_operations', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def start_spatial_operations():
    data = request.get_json()
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400

    logger.info(f"Starting spatial operations for session_id: {session_id}")

    # Start the spatial operations as a background task
    socketio.start_background_task(run_full_spatial_operations, engine, session_id, socketio)

    return jsonify({'session_id': session_id, 'status': 'STARTED'}), 202

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {'message': 'Connected to server.'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join')
def handle_join(data):
    try:
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            logger.info(f"Client {request.sid} joined room: {session_id}")
            emit('joined', {'message': f'Joined room {session_id}.'}, room=session_id)
        else:
            logger.warning(f"Join attempt without session_id from client {request.sid}")
            emit('error', {'message': 'session_id is required to join a room.'})
    except Exception as e:
        logger.error(f"Error in handle_join: {str(e)}")
        emit('error', {'message': 'An error occurred while joining the room.'})

@app.route('/output/<filename>')
def serve_output_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    logger.info("Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)