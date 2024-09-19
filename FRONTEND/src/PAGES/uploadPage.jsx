import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/navbar';
import ResultsPage from '../PAGES/resultsPage';
import shp from 'shpjs';

const dataTypes = [
  { key: 'area-of-interest', label: 'Area of Interest Shapefile' },
  { key: 'soils', label: 'Soils' },
  { key: 'geology', label: 'Geology' },
  { key: 'digitalElevationModel', label: 'Digital Elevation Model' },
  { key: 'protected-areas', label: 'Protected Areas' },
  { key: 'rivers', label: 'Rivers' },
  { key: 'roads', label: 'Roads' },
];

const DataTypeSection = ({
  dataType,
  data,
  setData,
  onUpload,
  isUploading,
  uploadSuccess,
}) => {
  const onDrop = useCallback(
    async (acceptedFiles) => {
      try {
        const filesObject = {};
        for (const file of acceptedFiles) {
          const extension = file.name.split('.').pop().toLowerCase();
          if (['shp', 'dbf', 'prj', 'cpg'].includes(extension)) {
            filesObject[extension] = await file.arrayBuffer();
          }
        }

        if (!filesObject.shp) {
          throw new Error('SHP file is required');
        }

        const geojson = await shp(filesObject);
        console.log('Files object:', Object.keys(filesObject));
        console.log('Converted GeoJSON:', geojson);
        console.log('GeoJSON type:', geojson.type);
        console.log(
          'Number of features:',
          geojson.features ? geojson.features.length : 'N/A'
        );
        setData((prevData) => ({
          ...prevData,
          [dataType.key]: {
            ...prevData[dataType.key],
            files: acceptedFiles,
            geojson: geojson,
          },
        }));
      } catch (error) {
        console.error('Error converting shapefile to GeoJSON:', error);
        alert('Error converting shapefile to GeoJSON. Please try again.');
      }
    },
    [dataType.key, setData]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: undefined,
    multiple: true,
  });

  return (
    <div className="mb-12">
      <h2 className="text-2xl font-semibold mb-4">{dataType.label}</h2>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Name:
        </label>
        <input
          type="text"
          value={data[dataType.key].name}
          onChange={(e) =>
            setData((prevData) => ({
              ...prevData,
              [dataType.key]: {
                ...prevData[dataType.key],
                name: e.target.value,
              },
            }))
          }
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
          required
        />
      </div>
      <div
        {...getRootProps()}
        className={`mt-4 p-8 border-2 border-dashed rounded-md cursor-pointer ${
          isDragActive ? 'border-indigo-600 bg-indigo-50' : 'border-gray-300'
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-center text-lg text-gray-600">
          {data[dataType.key].files
            ? `${data[dataType.key].files.length} files selected`
            : `Drag & drop .shp, .dbf, .prj, and .cpg files here, or click to select`}
        </p>
      </div>
      <button
        onClick={() => onUpload(dataType.key)}
        disabled={isUploading || uploadSuccess}
        className={`mt-4 w-full py-3 px-4 border border-transparent rounded-md text-lg font-medium text-white 
          ${
            isUploading
              ? 'bg-blue-400 cursor-not-allowed'
              : uploadSuccess
              ? 'bg-green-500 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
          }`}
      >
        {isUploading
          ? 'Uploading...'
          : uploadSuccess
          ? 'Uploaded Successfully'
          : 'Upload'}
      </button>
    </div>
  );
};

const Sidebar = ({ activeSection, setActiveSection, data }) => {
  const isComplete = (type) => {
    const section = data[type.key];
    return section.name && section.files;
  };

  return (
    <nav className="w-64 bg-gray-100 h-screen fixed left-0 top-16 p-4">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Data Types</h2>
      <ul>
        {dataTypes.map((type) => (
          <li key={type.key} className="mb-2">
            <button
              onClick={() => setActiveSection(type.key)}
              className={`w-full text-left p-2 rounded flex items-center justify-between ${
                activeSection === type.key
                  ? 'bg-blue-600 text-white'
                  : 'hover:bg-gray-200 text-gray-700'
              }`}
            >
              <span>{type.label}</span>
              {isComplete(type) && <span className="text-green-500">✓</span>}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
};

const UploadPage = () => {
  const [data, setData] = useState(
    dataTypes.reduce(
      (acc, type) => ({
        ...acc,
        [type.key]: { name: '', files: null, geojson: null },
      }),
      {}
    )
  );
  const [message, setMessage] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState({});
  const [activeSection, setActiveSection] = useState(dataTypes[0].key);
  const [activePage, setActivePage] = useState('upload');

  const handleUpload = async (dataTypeKey) => {
    const section = data[dataTypeKey];
    if (!section.name || !section.geojson) {
      setMessage(
        `Please complete all fields for ${dataTypeKey} before uploading.`
      );
      return;
    }

    setIsUploading(true);
    setMessage('Preparing upload...');

    let payload = {};

    if (dataTypeKey === 'soils') {
      payload = {
        name: section.name,
        features: section.geojson,
      };
    } else {
      payload = {
        name: section.name,
        geojson: section.geojson,
      };
    }

    let url = `http://localhost:3000/api/${dataTypeKey}`;
    console.log('Uploading data type:', dataTypeKey);
    console.log('Payload:', JSON.stringify(payload, null, 2));

    try {
      setMessage('Sending request to server...');
      const response = await axios.post(url, payload, {
        headers: { 'Content-Type': 'application/json' },
      });
      console.log('Server response:', response.data);
      setMessage(`${dataTypeKey} data uploaded successfully!`);
      setUploadSuccess((prev) => ({ ...prev, [dataTypeKey]: true }));

      // Move to the next section
      const currentIndex = dataTypes.findIndex(
        (type) => type.key === dataTypeKey
      );
      if (currentIndex < dataTypes.length - 1) {
        setActiveSection(dataTypes[currentIndex + 1].key);
      }
    } catch (error) {
      console.error('Upload error:', error);
      if (error.response) {
        console.error('Error response:', error.response.data);
      } else if (error.request) {
        console.error('No response received:', error.request);
      } else {
        console.error('Error setting up request:', error.message);
      }
      setMessage(`Error uploading ${dataTypeKey} data: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };
  const handleCleanDatabase = async () => {
    try {
      setMessage('Cleaning database...');
      const responses = await Promise.all([
        axios.delete('http://localhost:3000/api/rivers/deleteAll'),
        axios.delete('http://localhost:3000/api/protected-areas/deleteAll'),
        // Add more delete requests for other data types here
      ]);
      const totalDeleted = responses.reduce(
        (sum, response) => sum + response.data.count,
        0
      );
      setMessage(`Database cleaned. ${totalDeleted} items deleted.`);
      // Reset state
      setData(
        dataTypes.reduce(
          (acc, type) => ({
            ...acc,
            [type.key]: { name: '', files: null, geojson: null },
          }),
          {}
        )
      );
      setUploadSuccess({});
    } catch (error) {
      console.error('Error cleaning database:', error);
      setMessage(`Error cleaning database: ${error.message}`);
    }
  };
  return (
    <div className="bg-gray-50 min-h-screen">
      <Navbar activePage={activePage} setActivePage={setActivePage} />
      <div className="flex pt-16">
        <Sidebar
          activeSection={activeSection}
          setActiveSection={setActiveSection}
          data={data}
        />
        <div className="ml-64 flex-grow p-8">
          {activePage === 'upload' ? (
            <>
              <h1 className="text-3xl font-bold mb-8 text-center text-gray-800">
                Upload Geographical Data
              </h1>
              <button
                onClick={handleCleanDatabase}
                className="mb-8 py-2 px-4 bg-red-600 text-white rounded hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50"
              >
                Clean Database
              </button>
              {dataTypes.map((type) => (
                <div
                  key={type.key}
                  id={type.key}
                  className={activeSection === type.key ? '' : 'hidden'}
                >
                  <DataTypeSection
                    dataType={type}
                    data={data}
                    setData={setData}
                    onUpload={handleUpload}
                    isUploading={isUploading}
                    uploadSuccess={uploadSuccess[type.key]}
                  />
                </div>
              ))}
              {message && (
                <p
                  className={`mt-4 text-lg ${
                    message.includes('successfully') ||
                    message.includes('deleted')
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {message}
                </p>
              )}
            </>
          ) : (
            <ResultsPage />
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadPage;
