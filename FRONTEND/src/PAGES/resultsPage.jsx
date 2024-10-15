import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

const ResultsPage = () => {
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('Not Started');
  const [bufferImages, setBufferImages] = useState({});
  const socketRef = useRef(null);
  const sessionIdRef = useRef(`session_${Date.now()}`);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    console.log('Initializing socket connection');
    socketRef.current = io(API_BASE_URL, {
      transports: ['websocket'],
      upgrade: false,
    });

    socketRef.current.on('connect', () => {
      console.log('Connected to Socket.IO server');
      console.log('Socket ID:', socketRef.current.id);
      socketRef.current.emit('join', { session_id: sessionIdRef.current });
    });

    socketRef.current.on('connection_response', (data) => {
      console.log('Connection response:', data.message);
    });

    socketRef.current.on('joined', (data) => {
      console.log('Joined room:', data.message);
    });

    socketRef.current.on('progress_update', (data) => {
      console.log('Received progress update:', data);
      if (data.session_id === sessionIdRef.current) {
        setMessages((prevMessages) => [
          ...prevMessages,
          { id: Date.now(), text: data.message, status: 'completed' },
        ]);
      }
    });

    socketRef.current.on('task_error', (data) => {
      console.log('Received task error:', data);
      if (data.session_id === sessionIdRef.current) {
        setMessages((prevMessages) => [
          ...prevMessages,
          { id: Date.now(), text: data.message, status: 'error' },
        ]);
        setStatus('FAILED');
      }
    });

    socketRef.current.on('buffer_images', (data) => {
      console.log('Received buffer images:', data);
      if (data.session_id === sessionIdRef.current) {
        setBufferImages(data.images);
      }
    });

    socketRef.current.on('operation_completed', (data) => {
      console.log('Operation completed:', data);
      if (data.session_id === sessionIdRef.current) {
        setStatus('COMPLETED');
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: Date.now(),
            text: 'All operations completed successfully.',
            status: 'completed',
          },
        ]);
      }
    });

    socketRef.current.on('error', (data) => {
      console.log('Received error:', data);
      setError(data.message);
      setStatus('FAILED');
    });

    return () => {
      console.log('Cleaning up socket connection');
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    console.log('Messages updated:', messages);
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const startSpatialOperations = () => {
    setError(null);
    setMessages([]);
    setStatus('STARTING');
    setBufferImages({});

    axios
      .post(`${API_BASE_URL}/start_spatial_operations`, {
        session_id: sessionIdRef.current,
      })
      .then((response) => {
        console.log('Spatial operations started:', response.data);
        setStatus('RUNNING');
      })
      .catch((err) => {
        console.error('Error starting spatial operations:', err);
        setError(err.message);
        setStatus('FAILED');
      });
  };

  return (
    <div className="container mx-auto px-4 py-8 w-full">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="px-6 py-4 bg-blue-600">
          <h1 className="text-3xl font-bold text-white">
            Spatial Analysis Results
          </h1>
        </div>

        <div className="p-6">
          <button
            onClick={startSpatialOperations}
            className={`w-full mb-6 px-4 py-2 text-white font-semibold rounded ${
              status === 'RUNNING' || status === 'STARTING'
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600'
            }`}
            disabled={status === 'RUNNING' || status === 'STARTING'}
          >
            {status === 'RUNNING' || status === 'STARTING'
              ? 'Processing...'
              : 'Start analysis'}
          </button>

          {error && (
            <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-md">
              <strong>Error:</strong> {error}
            </div>
          )}

          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-2">Operation Status</h2>
            <div
              className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${
                status === 'COMPLETED'
                  ? 'bg-green-500 text-white'
                  : status === 'FAILED'
                  ? 'bg-red-500 text-white'
                  : 'bg-blue-500 text-white'
              }`}
            >
              {status}
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-4">Progress Messages</h2>
            {messages.length === 0 ? (
              <p className="text-gray-600 italic">
                No messages yet. Click the button to start spatial operations.
              </p>
            ) : (
              <ul className="space-y-2 max-h-96 overflow-y-auto border border-gray-200 p-4 rounded">
                {messages.map((message) => (
                  <li key={message.id} className="flex items-center">
                    {message.status === 'completed' ? (
                      <span className="flex-shrink-0 w-6 h-6 mr-3 bg-green-500 rounded-full flex items-center justify-center text-white">
                        &#10004;
                      </span>
                    ) : message.status === 'error' ? (
                      <span className="flex-shrink-0 w-6 h-6 mr-3 bg-red-500 rounded-full flex items-center justify-center text-white">
                        &#10006;
                      </span>
                    ) : (
                      <span className="flex-shrink-0 w-6 h-6 mr-3 bg-gray-300 rounded-full"></span>
                    )}
                    <span className="text-gray-800">{message.text}</span>
                  </li>
                ))}
                <div ref={messagesEndRef} />
              </ul>
            )}
          </div>

          {status === 'COMPLETED' && (
            <div className="mt-6">
              <h2 className="text-xl font-semibold mb-2">Generated Maps</h2>
              <div className="space-y-4">
                {Object.entries(bufferImages).map(([key, path]) => (
                  <div key={key}>
                    <h3 className="font-medium mb-1">{key} Buffer</h3>
                    <img
                      src={`${API_BASE_URL}/${path.replace(/\\/g, '/')}`}
                      alt={`${key} Buffer`}
                      className="w-full h-auto rounded-md shadow-md"
                    />
                    <a
                      href={`${API_BASE_URL}/${path.replace(/\\/g, '/')}`}
                      target="_blank"
                      download={`${key}_buffer.png`}
                      className="mt-2 inline-block px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                      Download
                    </a>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
