import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import axios from 'axios';
import { FaSearch, FaDownload } from 'react-icons/fa';

const API_BASE_URL = 'http://localhost:5000';

const ResultsPage = () => {
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('Not Started');
  const [bufferImages, setBufferImages] = useState({});
  const [suitabilityMaps, setSuitabilityMaps] = useState({});
  const [activeTab, setActiveTab] = useState('all');
  const [selectedMap, setSelectedMap] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [completionTime, setCompletionTime] = useState(null);
  const socketRef = useRef(null);
  const sessionIdRef = useRef(`session_${Date.now()}`);

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
        setError(data.message);
      }
    });

    socketRef.current.on('buffer_images', (data) => {
      console.log('Received buffer images:', data);
      if (data.session_id === sessionIdRef.current) {
        setBufferImages(data.images);
      }
    });

    socketRef.current.on('suitability_maps', (data) => {
      console.log('Received suitability maps:', data);
      if (data.session_id === sessionIdRef.current) {
        setSuitabilityMaps(data.maps);
      }
    });

    socketRef.current.on('operation_completed', (data) => {
      console.log('Operation completed:', data);
      if (data.session_id === sessionIdRef.current) {
        setStatus('COMPLETED');
        setCompletionTime(new Date().toLocaleString());
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
    });

    return () => {
      console.log('Cleaning up socket connection');
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const startSpatialOperations = () => {
    setError(null);
    setMessages([]);
    setStatus('STARTING');
    setBufferImages({});
    setSuitabilityMaps({});
    setCompletionTime(null);

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
      });
  };

  const renderMapThumbnail = (title, imagePath, type) => (
    <div className="bg-white rounded-lg shadow-md overflow-hidden transition-all duration-300 hover:shadow-xl hover:scale-105">
      <img
        src={`${API_BASE_URL}/${imagePath.replace(/\\/g, '/')}`}
        alt={title}
        className="w-full h-48 object-cover cursor-pointer"
        onClick={() => setSelectedMap({ title, imagePath })}
      />
      <div className="p-4">
        <h3 className="font-semibold text-lg mb-2">{title}</h3>
        <div className="flex justify-between">
          <button
            className="text-blue-500 hover:text-blue-700"
            onClick={() => setSelectedMap({ title, imagePath })}
          >
            View
          </button>
          <a
            href={`${API_BASE_URL}/${imagePath.replace(/\\/g, '/')}`}
            download={`${title.toLowerCase().replace(' ', '_')}.png`}
            className="text-green-500 hover:text-green-700"
          >
            Download
          </a>
        </div>
      </div>
    </div>
  );

  const filteredMaps = () => {
    const allMaps = { ...bufferImages, ...suitabilityMaps };
    const filteredByTab =
      activeTab === 'all'
        ? allMaps
        : activeTab === 'buffer'
        ? bufferImages
        : suitabilityMaps;

    return Object.entries(filteredByTab).filter(([key]) =>
      key.toLowerCase().includes(searchTerm.toLowerCase())
    );
  };

  const downloadAllMaps = () => {
    // Implement logic to download all maps
    console.log('Downloading all maps...');
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="px-6 py-4 bg-blue-600 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">
            Spatial Analysis Results
          </h1>
          <button
            onClick={startSpatialOperations}
            className="px-4 py-2 text-white font-semibold rounded bg-green-500 hover:bg-green-600"
          >
            Start Analysis
          </button>
        </div>

        <div className="p-6">
          <div className="mb-6 flex items-center">
            <div className="w-full">
              <h2 className="text-xl font-semibold mb-2">Status: {status}</h2>
              <div className="h-2 bg-gray-200 rounded-full">
                <div
                  className={`h-full rounded-full ${
                    status === 'COMPLETED' ? 'bg-green-500' : 'bg-blue-500'
                  }`}
                  style={{ width: status === 'COMPLETED' ? '100%' : '50%' }}
                ></div>
              </div>
            </div>
          </div>

          <div className="flex justify-between items-center mb-6">
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('all')}
                className={`px-4 py-2 rounded ${
                  activeTab === 'all' ? 'bg-blue-500 text-white' : 'bg-gray-200'
                }`}
              >
                All Maps
              </button>
              <button
                onClick={() => setActiveTab('buffer')}
                className={`px-4 py-2 rounded ${
                  activeTab === 'buffer'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200'
                }`}
              >
                Buffer Maps
              </button>
              <button
                onClick={() => setActiveTab('suitability')}
                className={`px-4 py-2 rounded ${
                  activeTab === 'suitability'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200'
                }`}
              >
                Suitability Maps
              </button>
            </div>
            <div className="relative">
              <input
                type="text"
                placeholder="Search maps..."
                className="pl-10 pr-4 py-2 border rounded-full"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredMaps().map(([key, path]) =>
              renderMapThumbnail(
                key,
                path,
                key.includes('Buffer') ? 'buffer' : 'suitability'
              )
            )}
          </div>

          {status === 'COMPLETED' && (
            <div className="mt-6 flex justify-end">
              <button
                onClick={downloadAllMaps}
                className="flex items-center px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                <FaDownload className="mr-2" /> Download All
              </button>
            </div>
          )}

          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-2">Progress Messages</h2>
            <div className="bg-gray-100 p-4 rounded-lg max-h-60 overflow-y-auto">
              {messages.map((message, index) => (
                <div key={index} className="flex items-center mb-2">
                  <span
                    className={`w-3 h-3 rounded-full mr-2 ${
                      message.status === 'completed'
                        ? 'bg-green-500'
                        : 'bg-yellow-500'
                    }`}
                  ></span>
                  <span>{message.text}</span>
                </div>
              ))}
            </div>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-100 text-red-700 rounded-md">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>
      </div>

      {selectedMap && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full">
            <div className="p-4 border-b flex justify-between items-center">
              <h2 className="text-2xl font-bold">{selectedMap.title}</h2>
              <button
                onClick={() => setSelectedMap(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                &times;
              </button>
            </div>
            <div className="p-4">
              <img
                src={`${API_BASE_URL}/${selectedMap.imagePath.replace(
                  /\\/g,
                  '/'
                )}`}
                alt={selectedMap.title}
                className="w-full h-auto"
              />
            </div>
            <div className="p-4 border-t flex justify-end">
              <a
                href={`${API_BASE_URL}/${selectedMap.imagePath.replace(
                  /\\/g,
                  '/'
                )}`}
                download={`${selectedMap.title
                  .toLowerCase()
                  .replace(' ', '_')}.png`}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 mr-2"
              >
                Download
              </a>
              <button
                onClick={() => setSelectedMap(null)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsPage;
