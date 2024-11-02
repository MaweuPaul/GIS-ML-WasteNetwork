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
  try {
    const { geometry, properties } = req.body;

    // Ensure we're working with a MultiPolygon
    const validGeometry = {
      type: 'MultiPolygon',
      coordinates: geometry.coordinates.map((polygon) => {
        return polygon.map((ring) => {
          // Ensure ring is closed
          const firstPoint = ring[0];
          const lastPoint = ring[ring.length - 1];
          if (
            firstPoint[0] !== lastPoint[0] ||
            firstPoint[1] !== lastPoint[1]
          ) {
            return [...ring, firstPoint];
          }
          return ring;
        });
      }),
    };

    // Insert with high precision geometry handling
    const [soil] = await prisma.$queryRaw`
      WITH valid_geom AS (
        SELECT ST_Multi(
          ST_SetSRID(
            ST_GeomFromGeoJSON(${JSON.stringify(validGeometry)}),
            4326
          )
        )::geometry(MultiPolygon, 4326) as geom
      )
      INSERT INTO "Soil" (
        "soilType",
        "geom",
        "properties",
        "geometryType",
        "coordinates",
        "bbox",
        "createdAt",
        "updatedAt"
      ) 
      SELECT
        ${properties.FAOSOIL}::text,
        geom,
        ${JSON.stringify(properties)}::jsonb,
        'MultiPolygon'::text,
        ${JSON.stringify(geometry.coordinates)}::jsonb,
        ${JSON.stringify(geometry.bbox)}::float[],
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
      FROM valid_geom
      RETURNING *;
    `;

    // Fetch the geometry with high precision
    const [geomResult] = await prisma.$queryRaw`
      SELECT ST_AsGeoJSON(geom, 15) as geom
      FROM "Soil"
      WHERE id = ${soil.id};
    `;

    const result = {
      ...soil,
      geom: JSON.parse(geomResult.geom),
    };

    res.status(201).json({
      message: 'Soil created successfully',
      soil: result,
    });
  } catch (error) {
    console.error('Error in createSoils:', error);
    res.status(500).json({
      message: 'Failed to create soil',
      error: error.message,
    });
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
