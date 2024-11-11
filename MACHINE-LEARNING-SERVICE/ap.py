from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS, cross_origin
from sqlalchemy import create_engine
import os
import logging
from datetime import datetime

from utils.spatial_operations import run_full_spatial_operations, perform_network_analysis

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
    async_mode='threading'
)

# Database connection setup
connection_params = {
    'dbname': 'wms',
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

@app.route('/output/<path:filename>')
@cross_origin(origin='*')
def serve_output_file(filename):
    """Serve any file from the output directory"""
    try:
        return send_from_directory('output', filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 404

@app.route('/get-network-map/<session_id>')
@cross_origin(origin='*')
def get_network_map(session_id):
    """Serve the network analysis map"""
    try:
        map_path = os.path.join(os.getcwd(), 'output', 'network_analysis_latest.html')
        if os.path.exists(map_path):
            return send_file(
                map_path,
                mimetype='text/html',
                as_attachment=False
            )
        else:
            return jsonify({'error': 'Map not found'}), 404
    except Exception as e:
        logger.error(f"Error serving network map: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-analysis-files/<session_id>')
@cross_origin(origin='*')
def get_analysis_files(session_id):
    """Get all analysis-related files"""
    try:
        output_dir = 'output'
        results = {
            'map': f'/get-network-map/{session_id}',
            'files': []
        }
        
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                file_info = {
                    'name': filename,
                    'url': f'/output/{filename}',
                    'type': get_file_type(filename),
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                }
                results['files'].append(file_info)
        
        results['files'].sort(key=lambda x: x['modified'], reverse=True)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting analysis files: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_file_type(filename):
    """Determine file type based on extension"""
    ext = filename.lower().split('.')[-1]
    type_mapping = {
        'html': 'map',
        'csv': 'data',
        'json': 'stats',
        'png': 'image',
        'jpg': 'image',
        'jpeg': 'image',
        'geojson': 'geodata'
    }
    return type_mapping.get(ext, 'other')

@app.route('/get-images/<session_id>')
@cross_origin(origin='*')
def get_images(session_id):
    """Get all images from output directory"""
    try:
        output_dir = 'output'
        images = []
        
        for filename in os.listdir(output_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(output_dir, filename)
                images.append({
                    'name': filename,
                    'url': f'/output/{filename}',
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        
        images.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'images': images})
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    logger.info("Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)