const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Soils
 */
const getSoils = async (req, res) => {
  try {
    const soils = await prisma.soil.findMany();
    res.json(soils);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch soils' });
  }
};

/**
 * Fetch a single Soil by ID
 */
const getSoil = async (req, res) => {
  const { id } = req.params;
  try {
    const soil = await prisma.soil.findUnique({ where: { id: Number(id) } });
    if (!soil) {
      return res.status(404).json({ message: 'Soil not found' });
    }
    res.json(soil);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch soil' });
  }
};

/**
 * Create multiple Soils
 */
const createSoils = async (req, res) => {
  const { features } = req.body;

  if (!Array.isArray(features)) {
    return res
      .status(400)
      .json({ message: 'Invalid input: Expected an array of features' });
  }

  try {
    const soilPromises = features.map((feature) => {
      const {
        type,
        geometry,
        properties: {
          objectId,
          featureId,
          gridcode,
          shapeLeng,
          shapeArea,
          soilType,
        },
      } = feature;

      const bboxArray = geometry.bbox ? `{${geometry.bbox.join(',')}}` : 'NULL';

      // Use raw SQL to handle geom field
      return prisma.$executeRaw`
        INSERT INTO "Soil" (
          "type",
          "geometryType",
          "coordinates",
          "bbox",
          "objectId",
          "featureId",
          "gridcode",
          "shapeLeng",
          "shapeArea",
          "soilType",
          "geom",
          "createdAt",
          "updatedAt"
        ) VALUES (
          ${type},
          ${geometry.type},
          ${JSON.stringify(geometry.coordinates)}::jsonb,
          ${bboxArray}::double precision[],
          ${objectId},
          ${featureId},
          ${gridcode},
          ${shapeLeng},
          ${shapeArea},
          ${soilType},
          ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(geometry)}), 4326),
          NOW(),
          NOW()
        ) RETURNING *;
      `;
    });

    const createdSoils = await prisma.$transaction(soilPromises);

    res.status(201).json({
      message: 'Soil data created',
      count: createdSoils.length,
      soils: createdSoils,
    });
  } catch (error) {
    console.error('Error creating soil data:', error);
    res
      .status(500)
      .json({ message: 'Failed to create soil data', error: error.message });
  }
};
/**
 * Update a Soil by ID
 */
const updateSoil = async (req, res) => {
  const { id } = req.params;
  const { name, description, geojson } = req.body;
  try {
    let parsedGeojson;
    if (typeof geojson === 'string') {
      parsedGeojson = JSON.parse(geojson);
    } else {
      parsedGeojson = geojson;
    }

    if (
      parsedGeojson.type !== 'FeatureCollection' ||
      !Array.isArray(parsedGeojson.features)
    ) {
      return res.status(400).json({ message: 'Invalid GeoJSON format' });
    }

    const feature = parsedGeojson.features[0];
    const { geometry, properties } = feature;

    const bboxArray = geometry.bbox ? `{${geometry.bbox.join(',')}}` : 'NULL';

    const updatedSoil = await prisma.$executeRaw`
      UPDATE "Soil"
      SET
        "name" = ${name},
        "description" = ${description},
        "geom" = ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
          geometry
        )}), 4326),
        "bbox" = ${bboxArray}::double precision[],
        "properties" = ${JSON.stringify(properties)}::jsonb,
        "updatedAt" = NOW()
      WHERE "id" = ${Number(id)}
      RETURNING *;
    `;

    if (updatedSoil.length === 0) {
      return res.status(404).json({ message: 'Soil not found' });
    }

    res.json(updatedSoil[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update soil data' });
  }
};

/**
 * Delete a Soil by ID
 */
const deleteSoil = async (req, res) => {
  const { id } = req.params;
  try {
    const deleted = await prisma.soil.delete({ where: { id: Number(id) } });
    res.json({ message: 'Soil deleted', deleted });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete soil' });
  }
};

/**
 * Delete all Soils
 */
const deleteAllSoils = async (req, res) => {
  try {
    const deletedCount = await prisma.soil.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Soils cleared successfully',
      count: deletedCount.count,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: 'Failed to clear soils',
      error: error.message,
    });
  }
};

module.exports = {
  getSoils,
  getSoil,
  createSoils,
  updateSoil,
  deleteSoil,
  deleteAllSoils,
};
