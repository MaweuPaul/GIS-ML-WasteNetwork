import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/navbar';
import ResultsPage from '../PAGES/resultsPage';

const dataTypes = [
  { key: 'soils', label: 'Soils' },
  { key: 'geology', label: 'Geology' },
  { key: 'digitalElevationModel', label: 'Digital Elevation Model' },
  { key: 'protectedAreas', label: 'Protected Areas' },
  { key: 'rivers', label: 'Rivers' },
  { key: 'roads', label: 'Roads' },
];

const DataTypeSection = ({ dataType, data, setData }) => {
  const onDrop = useCallback(
    (acceptedFiles) => {
      setData((prevData) => ({
        ...prevData,
        [dataType.key]: {
          ...prevData[dataType.key],
          file: acceptedFiles[0],
        },
      }));
    },
    [dataType.key, setData]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

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
      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Description:
        </label>
        <textarea
          value={data[dataType.key].description}
          onChange={(e) =>
            setData((prevData) => ({
              ...prevData,
              [dataType.key]: {
                ...prevData[dataType.key],
                description: e.target.value,
              },
            }))
          }
          rows="3"
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
          {data[dataType.key].file
            ? data[dataType.key].file.name
            : 'Drag & drop a file here, or click to select'}
        </p>
      </div>
    </div>
  );
};
const Sidebar = ({ activeSection, setActiveSection, data }) => {
  const isComplete = (type) => {
    const section = data[type.key];
    return section.name && section.description && section.file;
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
        [type.key]: { name: '', description: '', file: null },
      }),
      {}
    )
  );
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeSection, setActiveSection] = useState(dataTypes[0].key);
  const [activePage, setActivePage] = useState('upload');

  const handleUpload = async () => {
    const incompleteSection = dataTypes.find((type) => {
      const section = data[type.key];
      return !section.name || !section.description || !section.file;
    });

    if (incompleteSection) {
      setActiveSection(incompleteSection.key);
      setMessage(
        `Please complete all fields for ${incompleteSection.label} before proceeding.`
      );
      return;
    }

    setIsLoading(true);
    setMessage('');

    const uploadPromises = dataTypes.map((type) => {
      const formData = new FormData();
      formData.append('file', data[type.key].file);
      formData.append('name', data[type.key].name);
      formData.append('description', data[type.key].description);

      return axios.post(`http://localhost:3000/${type.key}s`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    });

    try {
      await Promise.all(uploadPromises);
      setMessage('All data uploaded successfully!');
      setData(
        dataTypes.reduce(
          (acc, type) => ({
            ...acc,
            [type.key]: { name: '', description: '', file: null },
          }),
          {}
        )
      );
    } catch (error) {
      setMessage('Error uploading one or more files. Please try again.');
      console.error('Upload error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNext = () => {
    const currentIndex = dataTypes.findIndex(
      (type) => type.key === activeSection
    );
    if (currentIndex < dataTypes.length - 1) {
      setActiveSection(dataTypes[currentIndex + 1].key);
    }
  };

  const isLastSection = activeSection === dataTypes[dataTypes.length - 1].key;

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
                  />
                </div>
              ))}
              <button
                onClick={isLastSection ? handleUpload : handleNext}
                disabled={isLoading}
                className={`w-full py-3 px-4 border border-transparent rounded-md text-lg font-medium text-white ${
                  isLoading
                    ? 'bg-blue-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
                }`}
              >
                {isLoading
                  ? 'Processing...'
                  : isLastSection
                  ? 'Upload All Data'
                  : 'Next'}
              </button>
              {message && (
                <p
                  className={`mt-4 text-lg ${
                    message.includes('successfully')
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
