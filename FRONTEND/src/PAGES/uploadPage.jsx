import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Navbar from '../components/Navbar';
import ResultsPage from '../pages/ResultsPage';
import axios from 'axios';
import shp from 'shpjs';
import MapVisualization from '../components/MapVisualizer';

const dataTypes = [
  { key: 'area-of-interest', label: 'Area of Interest Shapefile' },
  { key: 'soils', label: 'Soils' },
  { key: 'geology', label: 'Geology' },
  { key: 'digitalElevationModel', label: 'Digital Elevation Model' },
  { key: 'protected-areas', label: 'Protected Areas' },
  { key: 'rivers', label: 'Rivers' },
  { key: 'roads', label: 'Roads' },
  { key: 'settlement', label: 'Settlement' },
  { key: 'land-use-raster', label: 'Land Use Raster' }, // Retained for upload
];

const DataTypeSection = ({
  dataType,
  data,
  setData,
  onUpload,
  isUploading,
  uploadSuccess,
  setLayersData,
}) => {
  const onDrop = useCallback(
    async (acceptedFiles) => {
      try {
        if (dataType.key === 'land-use-raster') {
          // Handle Land Use Raster upload
          const file = acceptedFiles[0];
          if (file) {
            setData((prevData) => ({
              ...prevData,
              [dataType.key]: {
                ...prevData[dataType.key],
                file: file,
              },
            }));
          }
        } else {
          // Handle Shapefile uploads for vector data
          const filesObject = {};
          const requiredExtensions = ['shp', 'dbf', 'prj'];
          const optionalExtensions = ['cpg'];
          const missingRequired = [];

          for (const file of acceptedFiles) {
            const extension = file.name.split('.').pop().toLowerCase();
            if (
              [...requiredExtensions, ...optionalExtensions].includes(extension)
            ) {
              filesObject[extension] = await file.arrayBuffer();
            }
          }

          for (const ext of requiredExtensions) {
            if (!filesObject[ext]) {
              missingRequired.push(ext.toUpperCase());
            }
          }

          if (missingRequired.length > 0) {
            throw new Error(
              `Missing required files: ${missingRequired.join(', ')}`
            );
          }

          const geojson = await shp(filesObject);
          console.log('Converted GeoJSON:', geojson);

          setData((prevData) => ({
            ...prevData,
            [dataType.key]: {
              ...prevData[dataType.key],
              files: acceptedFiles,
              geojson: geojson,
            },
          }));

          setLayersData((prevData) => ({
            ...prevData,
            [dataType.key]: geojson,
          }));
        }
      } catch (error) {
        console.error('Error processing file:', error);
        alert(`Error processing file: ${error.message}`);
      }
    },
    [dataType.key, setData, setLayersData]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept:
      dataType.key === 'land-use-raster'
        ? { 'image/tiff': ['.tif', '.tiff'] }
        : {
            'application/x-shapefile': ['.shp'],
            'application/octet-stream': ['.dbf'],
            'application/json': ['.prj'],
            'application/x-cpg': ['.cpg'],
          },
    multiple: dataType.key !== 'land-use-raster',
  });

  return (
    <div className="mb-12">
      <h2 className="text-2xl font-semibold mb-4">{dataType.label}</h2>
      {dataType.key !== 'land-use-raster' ? (
        <>
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
              isDragActive
                ? 'border-indigo-600 bg-indigo-50'
                : 'border-gray-300'
            }`}
          >
            <input {...getInputProps()} />
            <p className="text-center text-lg text-gray-600">
              {data[dataType.key].files
                ? `${data[dataType.key].files.length} files selected`
                : `Drag & drop .shp, .dbf, and .prj files here (optional: .cpg), or click to select`}
            </p>
          </div>
        </>
      ) : (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description:
            </label>
            <input
              type="text"
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
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
              required
            />
          </div>
          <div
            {...getRootProps()}
            className={`mt-4 p-8 border-2 border-dashed rounded-md cursor-pointer ${
              isDragActive
                ? 'border-indigo-600 bg-indigo-50'
                : 'border-gray-300'
            }`}
          >
            <input {...getInputProps()} />
            <p className="text-center text-lg text-gray-600">
              {data[dataType.key].file
                ? `${data[dataType.key].file.name} selected`
                : `Drag & drop a .tif or .tiff file here, or click to select`}
            </p>
          </div>
        </>
      )}
      <button
        onClick={() => onUpload(dataType.key)}
        disabled={
          isUploading ||
          (dataType.key === 'land-use-raster'
            ? !data[dataType.key].file
            : !data[dataType.key].name || !data[dataType.key].geojson)
        }
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
    if (type.key === 'land-use-raster') {
      return data[type.key].file;
    }
    const section = data[type.key];
    return section.name && section.geojson;
  };

  return (
    <nav className="w-64 bg-gray-100 h-screen fixed left-0 top-16 p-4 overflow-auto">
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
        [type.key]:
          type.key === 'land-use-raster'
            ? { description: '', file: null }
            : { name: '', files: null, geojson: null },
      }),
      {}
    )
  );
  const [message, setMessage] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState({});
  const [activeSection, setActiveSection] = useState(dataTypes[0].key);
  const [activePage, setActivePage] = useState('upload');
  const [layersData, setLayersData] = useState(
    dataTypes.reduce(
      (acc, type) => ({
        ...acc,
        [type.key]: null,
      }),
      {}
    )
  );

  const handleUpload = async (dataTypeKey) => {
    const section = data[dataTypeKey];
    if (
      (dataTypeKey !== 'land-use-raster' &&
        (!section.name || !section.geojson)) ||
      (dataTypeKey === 'land-use-raster' && !section.file)
    ) {
      setMessage(
        `Please complete all fields for ${
          dataTypeKey === 'land-use-raster'
            ? 'Land Use Raster'
            : dataTypes.find((type) => type.key === dataTypeKey).label
        } before uploading.`
      );
      return;
    }

    setIsUploading(true);
    setMessage('Preparing upload...');

    try {
      switch (dataTypeKey) {
        case 'digitalElevationModel':
          await uploadDEM(section);
          break;

        case 'soils':
          await uploadSoils(section);
          break;

        case 'area-of-interest':
          await uploadAreaOfInterest(section);
          break;

        case 'land-use-raster':
          await uploadLandUseRaster(section, dataTypeKey); // Pass dataTypeKey
          break;

        case 'settlement':
          await uploadSettlement(section);
          break;

        default:
          await uploadGeneric(dataTypeKey, section);
      }

      setUploadSuccess((prev) => ({ ...prev, [dataTypeKey]: true }));
      setMessage(`${dataTypeKey} data uploaded successfully!`);
    } catch (error) {
      console.error('Upload error:', error);
      setMessage(`Error uploading ${dataTypeKey} data: ${error.message}`);
    } finally {
      setIsUploading(false);
      const currentIndex = dataTypes.findIndex(
        (type) => type.key === dataTypeKey
      );
      if (currentIndex < dataTypes.length - 1) {
        setActiveSection(dataTypes[currentIndex + 1].key);
      }
    }
  };

  const uploadDEM = async (section) => {
    const features = Array.isArray(section.geojson.features)
      ? section.geojson.features
      : [section.geojson];

    const modifiedFeatures = features.map((feature) => {
      return {
        name: section.name,
        bbox: feature.geometry.bbox || [],
        geometryType: feature.geometry.type,
        geometry: feature.geometry,
        coordinates: feature.geometry.coordinates,
        elevation: feature.properties.gridcode,
      };
    });

    const chunkSize = 10;
    const totalChunks = Math.ceil(modifiedFeatures.length / chunkSize);

    for (let i = 0; i < modifiedFeatures.length; i += chunkSize) {
      const chunk = modifiedFeatures.slice(i, i + chunkSize);
      const payload = { features: chunk };

      setMessage(
        `Sending chunk ${Math.floor(i / chunkSize) + 1} of ${totalChunks}...`
      );
      const response = await axios.post(
        'http://localhost:3000/api/digital-elevation-models',
        payload,
        {
          headers: { 'Content-Type': 'application/json' },
        }
      );
      console.log(
        `Chunk ${Math.floor(i / chunkSize) + 1} response:`,
        response.data
      );
    }

    setMessage('DEM data uploaded successfully!');
  };

  const uploadSoils = async (section) => {
    const features = Array.isArray(section.geojson.features)
      ? section.geojson.features
      : [section.geojson];
    const modifiedFeatures = features.map((feature) => ({
      type: feature.type,
      geometry: {
        type: feature.geometry.type,
        coordinates: feature.geometry.coordinates,
        bbox: feature.geometry.bbox,
      },
      properties: {
        objectId: feature.properties.OBJECTID || feature.properties.objectId,
        featureId: feature.properties.Id || feature.properties.featureId,
        gridcode: feature.properties.gridcode,
        shapeLeng:
          feature.properties.Shape_Leng || feature.properties.shapeLeng,
        shapeArea:
          feature.properties.Shape_Area || feature.properties.shapeArea,
        soilType: feature.properties.soil_type || feature.properties.soilType,
      },
    }));
    const payload = { features: modifiedFeatures };

    setMessage('Uploading Soils data...');
    const response = await axios.post(
      'http://localhost:3000/api/soils',
      payload,
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );
    console.log('Soils upload response:', response.data);
  };

  const uploadAreaOfInterest = async (section) => {
    const feature =
      section.geojson.type === 'Feature'
        ? section.geojson
        : section.geojson.features[0];

    const payload = {
      feature: {
        type: feature.type,
        geometry: {
          type: feature.geometry.type,
          coordinates: feature.geometry.coordinates,
          bbox: feature.geometry.bbox,
        },
        properties: {
          ...feature.properties,
          NAME_2: section.name,
        },
      },
    };

    setMessage('Uploading Area of Interest...');
    const response = await axios.post(
      'http://localhost:3000/api/area-of-interest',
      payload,
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );
    console.log('Area of Interest upload response:', response.data);
  };

  const uploadLandUseRaster = async (section, dataTypeKey) => {
    const formData = new FormData();
    formData.append('file', section.file);
    formData.append('description', section.description);

    setMessage('Uploading Land Use Raster data...');
    try {
      const response = await axios.post(
        'http://localhost:3000/api/land-use-raster',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      console.log('Land Use Raster upload response:', response.data);

      // Optionally, mark upload as successful
      setUploadSuccess((prev) => ({ ...prev, [dataTypeKey]: true }));
      setMessage('Land Use Raster data uploaded successfully!');

      // Do NOT add to layersData to exclude from visualization
      // If you need to store it elsewhere, handle accordingly
    } catch (error) {
      console.error('Land Use Raster upload error:', error);
      setMessage(`Error uploading Land Use Raster data: ${error.message}`);
    }
  };

  const uploadLandUse = async (section) => {
    const features = Array.isArray(section.geojson.features)
      ? section.geojson.features
      : [section.geojson];
    const modifiedFeatures = features.map((feature) => ({
      type: feature.type,
      geometry: {
        type: feature.geometry.type,
        coordinates: feature.geometry.coordinates,
        bbox: feature.geometry.bbox,
      },
      properties: {
        ...feature.properties,
        landUseType:
          feature.properties.land_use_type || feature.properties.landUseType,
      },
    }));
    const payload = { features: modifiedFeatures };

    setMessage('Uploading Land Use data...');
    const response = await axios.post(
      'http://localhost:3000/api/land-use',
      payload,
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );
    console.log('Land Use upload response:', response.data);
  };

  const uploadSettlement = async (section) => {
    console.log('Section data:', section);

    const features = Array.isArray(section.geojson.features)
      ? section.geojson.features
      : [section.geojson];

    console.log('Number of features:', features.length);

    const modifiedFeatures = features.map((feature, index) => {
      console.log(`Feature ${index} geometry:`, feature.geometry);
      if (!feature.geometry) {
        console.error(`Feature ${index} is missing geometry:`, feature);
      }
      return {
        name:
          feature.properties.settlement_name ||
          feature.properties.settlementName ||
          section.name ||
          `Settlement ${index}`,
        type: feature.type,
        geometry: feature.geometry,
        properties: {
          ...feature.properties,
          population: feature.properties.population,
        },
      };
    });

    const chunkSize = 20;
    const totalChunks = Math.ceil(modifiedFeatures.length / chunkSize);

    for (let i = 0; i < modifiedFeatures.length; i += chunkSize) {
      const chunk = modifiedFeatures.slice(i, i + chunkSize);
      const payload = chunk; // Send the chunk directly, not wrapped in a 'features' object

      console.log(
        `Chunk ${Math.floor(i / chunkSize) + 1} payload:`,
        JSON.stringify(payload, null, 2)
      );

      setMessage(
        `Uploading Settlement data: chunk ${
          Math.floor(i / chunkSize) + 1
        } of ${totalChunks}...`
      );

      try {
        const response = await axios.post(
          'http://localhost:3000/api/settlement',
          payload,
          {
            headers: { 'Content-Type': 'application/json' },
          }
        );
        console.log(
          `Settlement chunk ${Math.floor(i / chunkSize) + 1} upload response:`,
          response.data
        );
      } catch (error) {
        console.error(
          `Error uploading Settlement chunk ${Math.floor(i / chunkSize) + 1}:`,
          error.response ? error.response.data : error.message
        );
        throw new Error(
          `Failed to upload Settlement chunk ${
            Math.floor(i / chunkSize) + 1
          }: ${error.message}`
        );
      }
    }

    setMessage('Settlement data uploaded successfully!');
  };

  const uploadGeneric = async (dataTypeKey, section) => {
    const payload = {
      name: section.name,
      geojson: section.geojson,
    };

    const url = `http://localhost:3000/api/${dataTypeKey}`;
    console.log('Uploading data type:', dataTypeKey);
    console.log('Payload:', JSON.stringify(payload, null, 2));

    const response = await axios.post(url, payload, {
      headers: { 'Content-Type': 'application/json' },
    });
    console.log('Server response:', response.data);
  };

  const handleCleanDatabase = async () => {
    try {
      setMessage('Checking database status...');

      // First, check if the database is empty
      const checkResponse = await axios.get(
        'http://localhost:3000/api/database/check'
      );

      if (checkResponse.data.isEmpty) {
        setMessage('Database is already empty. No cleaning necessary.');
        return;
      }

      setMessage('Cleaning database...');
      const endpoints = [
        'rivers',
        'protected-areas',
        'soils',
        'area-of-interest',
        'digital-elevation-models',
        'roads',
        'land-use-raster',
        'geology',
        'settlement',
      ];

      const responses = await Promise.all(
        endpoints.map((endpoint) =>
          axios
            .delete(`http://localhost:3000/api/${endpoint}/deleteAll`)
            .catch((error) => {
              console.error(
                `Error deleting ${endpoint}:`,
                error.response || error
              );
              return { data: { success: false, error: error.message } };
            })
        )
      );

      let totalDeleted = 0;
      const details = [];

      responses.forEach((response, index) => {
        const dataType = endpoints[index]
          .replace(/-/g, ' ')
          .replace(/\b\w/g, (l) => l.toUpperCase());
        console.log(`Response for ${dataType}:`, response);

        if (response.data && response.data.success) {
          const count =
            response.data.count !== undefined ? response.data.count : 'unknown';
          details.push(
            `${dataType}: Cleared successfully (${count} items deleted)`
          );
          if (typeof response.data.count === 'number') {
            totalDeleted += response.data.count;
          }
        } else {
          const errorMessage = response.data
            ? response.data.error || response.data.message
            : 'Unknown error';
          details.push(`${dataType}: Clear operation failed (${errorMessage})`);
        }
      });

      const deletionMessage =
        totalDeleted > 0
          ? `Database cleaned. ${totalDeleted} items deleted.`
          : 'Database cleaned. No items were deleted.';

      setMessage(`${deletionMessage}\n\nDetails:\n${details.join('\n')}`);

      // Reset state
      setData(
        dataTypes.reduce(
          (acc, type) => ({
            ...acc,
            [type.key]:
              type.key === 'land-use-raster'
                ? { description: '', file: null }
                : { name: '', files: null, geojson: null },
          }),
          {}
        )
      );
      setUploadSuccess({});
      setLayersData(
        dataTypes.reduce(
          (acc, type) => ({
            ...acc,
            [type.key]: null,
          }),
          {}
        )
      );
    } catch (error) {
      console.error('Error cleaning database:', error);
      setMessage(`Error cleaning database: ${error.message}`);
    }
  };

  return (
    <div className="bg-gray-50 min-h-screen">
      <Navbar
        activePage={activePage}
        setActivePage={setActivePage}
        handleCleanDatabase={handleCleanDatabase}
      />
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
                    onUpload={handleUpload}
                    isUploading={isUploading}
                    uploadSuccess={uploadSuccess[type.key]}
                    setLayersData={setLayersData}
                  />
                </div>
              ))}
              {message && (
                <p
                  className={`mt-4 text-lg whitespace-pre-wrap ${
                    message.includes('successfully') ||
                    message.includes('deleted')
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {message}
                </p>
              )}
              <div className="mt-8">
                <h2 className="text-2xl font-semibold mb-4">
                  Data Visualization
                </h2>
                <MapVisualization layersData={layersData} />
              </div>
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
