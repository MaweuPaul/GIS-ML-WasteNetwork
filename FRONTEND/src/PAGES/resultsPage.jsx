import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import axios from 'axios';
import { FaSearch, FaDownload, FaFileExcel, FaFileImage } from 'react-icons/fa';

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
  const [startTime, setStartTime] = useState(null);
  const [excelReportPath, setExcelReportPath] = useState(null);
  const [tiffFiles, setTiffFiles] = useState({});
  const [networkAnalysisUrl, setNetworkAnalysisUrl] = useState(null);
  const [isNetworkAnalysisLoading, setIsNetworkAnalysisLoading] =
    useState(false);

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

    socketRef.current.on('progress_update', (data) => {
      console.log('Received progress update:', data);
      if (data.session_id === sessionIdRef.current) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: Date.now(),
            text: data.message,
            status: 'completed',
            elapsedTime: getElapsedTime(),
          },
        ]);
      }
    });

    socketRef.current.on('task_error', (data) => {
      console.log('Received task error:', data);
      if (data.session_id === sessionIdRef.current) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: Date.now(),
            text: data.message,
            status: 'error',
            elapsedTime: getElapsedTime(),
          },
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
            elapsedTime: getElapsedTime(),
          },
        ]);
      }
    });

    socketRef.current.on('error', (data) => {
      console.log('Received error:', data);
      setError(data.message);
    });

    socketRef.current.on('excel_report', (data) => {
      console.log('Received Excel report path:', data);
      if (data.session_id === sessionIdRef.current) {
        setExcelReportPath(data.path);
      }
    });

    socketRef.current.on('tiff_files', (data) => {
      console.log('Received TIFF files:', data);
      if (data.session_id === sessionIdRef.current) {
        setTiffFiles(data.files);
      }
    });

    return () => {
      console.log('Cleaning up socket connection');
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    let timer;
    if (startTime && status !== 'COMPLETED') {
      timer = setInterval(() => {
        setMessages((prevMessages) => {
          const newMessages = [...prevMessages];
          if (newMessages.length > 0) {
            newMessages[newMessages.length - 1] = {
              ...newMessages[newMessages.length - 1],
              elapsedTime: getElapsedTime(),
            };
          }
          return newMessages;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [startTime, status]);

  const getElapsedTime = () => {
    if (!startTime) return 'Not started';
    const elapsed = Date.now() - startTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    return `${hours.toString().padStart(2, '0')}:${(minutes % 60)
      .toString()
      .padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
  };

  const startSpatialOperations = () => {
    setError(null);
    setMessages([]);
    setStatus('STARTING');
    setBufferImages({});
    setSuitabilityMaps({});
    setCompletionTime(null);
    setStartTime(Date.now());

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
  useEffect(() => {
    if (!socketRef.current) return;

    socketRef.current.on('network_analysis_complete', (data) => {
      if (data.session_id === sessionIdRef.current) {
        // Using port 5000 for Flask API
        const url = `${API_BASE_URL}/output/network_analysis_session_${data.session_id}_latest.html`;
        setNetworkAnalysisUrl(url);
        setIsNetworkAnalysisLoading(false);
      }
    });

    return () => {
      socketRef.current.off('network_analysis_complete');
    };
  }, []);

  const loadNetworkAnalysis = async () => {
    setIsNetworkAnalysisLoading(true);
    try {
      // Using port 5000 for Flask API
      const url = `${API_BASE_URL}/output/network_analysis_session_${sessionIdRef.current}_latest.html`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Network analysis not available');
      }

      setNetworkAnalysisUrl(url);
      setError(null);
    } catch (error) {
      setError('Failed to load network analysis results');
      console.error('Network analysis error:', error);
      setNetworkAnalysisUrl(null);
    } finally {
      setIsNetworkAnalysisLoading(false);
    }
  };
  useEffect(() => {
    if (status === 'COMPLETED') {
      loadNetworkAnalysis();
    }
  }, [status]);

  const renderMapThumbnail = (title, imagePath, type) => (
    <div className="bg-white rounded-lg shadow-md overflow-hidden transition-all duration-300 hover:shadow-xl hover:scale-105">
      <img
        src={`${API_BASE_URL}/${imagePath}`}
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
            target="_blank"
            onClick={(e) => {
              e.preventDefault();
              downloadImage(
                `${API_BASE_URL}/${imagePath}`,
                `${title.toLowerCase().replace(' ', '_')}.png`
              );
            }}
            className="text-green-500 hover:text-green-700 cursor-pointer"
          >
            Download PNG
          </a>
          {tiffFiles[title] && (
            <a
              target="_blank"
              href={`${API_BASE_URL}/${tiffFiles[title]}`}
              download={`${title.toLowerCase().replace(' ', '_')}.tif`}
              className="text-purple-500 hover:text-purple-700"
            >
              Download TIFF
            </a>
          )}
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

  const downloadImage = (imageUrl, fileName, width = 1200, height = 1000) => {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = imageUrl;

    img.onload = function () {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = width;
      canvas.height = height;

      ctx.drawImage(img, 0, 0, width, height);

      const downloadLink = document.createElement('a');
      downloadLink.download = fileName;
      downloadLink.href = canvas.toDataURL('image/png');
      downloadLink.click();
    };
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="px-6 py-4 bg-blue-600 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">Analysis Results</h1>
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
              <div className="flex justify-between mb-2">
                <h2 className="text-xl font-semibold">Status: {status}</h2>
                {completionTime && (
                  <span className="text-green-600">
                    Completed at: {completionTime}
                  </span>
                )}
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    status === 'COMPLETED'
                      ? 'bg-green-500'
                      : status === 'RUNNING'
                      ? 'bg-blue-500'
                      : 'bg-gray-400'
                  }`}
                  style={{
                    width:
                      status === 'COMPLETED'
                        ? '100%'
                        : status === 'RUNNING'
                        ? '50%'
                        : '0%',
                    transition: 'width 0.5s ease-in-out',
                  }}
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
              <button
                onClick={() => setActiveTab('network')}
                className={`px-4 py-2 rounded ${
                  activeTab === 'network'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200'
                }`}
              >
                Network Analysis
              </button>
            </div>

            <div className="flex items-center">
              <div className="relative mr-4">
                <input
                  type="text"
                  placeholder="Search maps..."
                  className="pl-10 pr-4 py-2 border rounded-full"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              </div>
              {excelReportPath && (
                <a
                  target="_blank"
                  href={`${API_BASE_URL}/${excelReportPath}`}
                  download="analysis_report.xlsx"
                  className="flex items-center px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  <FaFileExcel className="mr-2" /> Download Excel Report
                </a>
              )}
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
                <FaDownload className="mr-2" /> Download All Maps
              </button>
            </div>
          )}

          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-2 flex items-center">
              Progress Messages
              {status === 'RUNNING' && (
                <span className="ml-2 text-sm text-blue-500">
                  Time Elapsed: {getElapsedTime()}
                </span>
              )}
            </h2>
            <div className="bg-gray-100 p-4 rounded-lg max-h-[400px] overflow-y-auto">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex items-center mb-2 p-2 rounded ${
                    message.status === 'completed'
                      ? 'bg-green-50'
                      : message.status === 'error'
                      ? 'bg-red-50'
                      : 'bg-white'
                  }`}
                >
                  <span
                    className={`w-3 h-3 rounded-full mr-2 flex-shrink-0 ${
                      message.status === 'completed'
                        ? 'bg-green-500'
                        : message.status === 'error'
                        ? 'bg-red-500'
                        : 'bg-yellow-500'
                    }`}
                  ></span>
                  <span className="flex-grow">{message.text}</span>
                  {message.elapsedTime && (
                    <span className="ml-4 text-sm text-gray-500">
                      {message.elapsedTime}
                    </span>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-100 text-red-700 rounded-md">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>
      </div>
      {activeTab === 'network' && (
        <div className="mt-6">
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="p-4 border-b">
              <h3 className="text-xl font-semibold">
                Network Analysis Results
              </h3>
            </div>
            {isNetworkAnalysisLoading ? (
              <div className="p-8 text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                <p className="mt-4 text-gray-500">
                  Loading network analysis...
                </p>
              </div>
            ) : networkAnalysisUrl ? (
              <div className="h-[800px] w-full">
                <iframe
                  src={networkAnalysisUrl}
                  className="w-full h-full border-none"
                  title="Network Analysis"
                />
              </div>
            ) : (
              <div className="p-8 text-center text-gray-500">
                {error ? (
                  <div className="text-red-500">
                    <p>Error loading network analysis:</p>
                    <p>{error}</p>
                    <button
                      onClick={loadNetworkAnalysis}
                      className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                      Retry
                    </button>
                  </div>
                ) : (
                  'Network analysis results will appear here when the analysis is complete.'
                )}
              </div>
            )}
          </div>
        </div>
      )}
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
                src={`${API_BASE_URL}/${selectedMap.imagePath}`}
                alt={selectedMap.title}
                className="w-full h-auto"
              />
            </div>
            <div className="p-4 border-t flex justify-end">
              <a
                target="_blank"
                href={`${API_BASE_URL}/${selectedMap.imagePath}`}
                onClick={(e) => {
                  e.preventDefault();
                  downloadImage(
                    `${API_BASE_URL}/${imagePath}`,
                    `${title.toLowerCase().replace(' ', '_')}.png`
                  );
                }}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 mr-2"
              >
                <FaFileImage className="mr-2 inline" />
                Download PNG
              </a>
              {tiffFiles[selectedMap.title] && (
                <a
                  href={`${API_BASE_URL}/${tiffFiles[selectedMap.title]}`}
                  download={`${selectedMap.title
                    .toLowerCase()
                    .replace(' ', '_')}.tif`}
                  className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 mr-2"
                >
                  <FaFileImage className="mr-2 inline" />
                  Download TIFF
                </a>
              )}
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
