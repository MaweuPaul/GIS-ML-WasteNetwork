import React, { useState, useEffect } from 'react';
import ShpToGeoJson from 'shp-to-geojson/dist/shp-to-geojson.browser.js';

const ShapefileToGeoJson = ({ filePath, onConversionComplete }) => {
  const [geoJsonData, setGeoJsonData] = useState(null);

  useEffect(() => {
    const convertShapefile = async () => {
      try {
        const shp = new ShpToGeoJson({
          filePath: filePath,
        });

        const featureCollection = await shp.getGeoJson();
        setGeoJsonData(featureCollection);

        // If you need to process features individually
        const features = [];
        const stream = shp.streamGeoJsonFeatures();
        let featureIterator = await stream.next();
        while (!featureIterator.done) {
          features.push(featureIterator.value);
          featureIterator = await stream.next();
        }

        // Call the callback function with the converted data
        onConversionComplete({
          featureCollection: featureCollection,
          features: features,
        });
      } catch (error) {
        console.error('Error converting shapefile to GeoJSON:', error);
      }
    };

    convertShapefile();
  }, [filePath, onConversionComplete]);

  return (
    <div>
      {geoJsonData
        ? 'Shapefile converted to GeoJSON'
        : 'Converting shapefile...'}
    </div>
  );
};

export default ShapefileToGeoJson;
