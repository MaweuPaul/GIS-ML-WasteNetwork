const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Roads
 */
const getRoads = async (req, res) => {
  try {
    const roads = await prisma.road.findMany();
    res.json(roads);
  } catch (error) {
    console.error('Error fetching roads:', error);
    res.status(500).json({ message: 'Failed to fetch roads' });
  }
};

/**
 * Fetch a single Road by ID
 */
const getRoad = async (req, res) => {
  const { id } = req.params;
  try {
    const road = await prisma.road.findUnique({ where: { id: Number(id) } });
    if (!road) {
      return res.status(404).json({ message: 'Road not found' });
    }
    res.json(road);
  } catch (error) {
    console.error('Error fetching road:', error);
    res.status(500).json({ message: 'Failed to fetch road' });
  }
};

/**
 * Create multiple Roads from GeoJSON
 */
const createRoad = async (req, res) => {
  const { geojson } = req.body;

  if (!geojson || !geojson.features || !Array.isArray(geojson.features)) {
    return res.status(400).json({ message: 'Invalid GeoJSON data' });
  }

  try {
    const createdRoads = await prisma.$transaction(
      geojson.features
        .map((feature) => {
          let { type, coordinates, bbox } = feature.geometry;
          const { name, ...otherProperties } = feature.properties;
          // Convert MultiLineString to LineString if it contains only one line
          if (type === 'MultiLineString' && coordinates.length === 1) {
            type = 'LineString';
            coordinates = coordinates[0];
          }
          if (type !== 'LineString') {
            return null;
          }
          const bboxArray = bbox ? `{${bbox.join(',')}}` : 'NULL';
          return prisma.$executeRaw`
          INSERT INTO "Road" (
            "name",
            "type",
            "coordinates",
            "bbox",
            "geom",
            "properties",
            "createdAt",
            "updatedAt"
          ) VALUES (
            ${name || 'Unnamed Road'},
            ${type},
            ${JSON.stringify(coordinates)}::jsonb,
            ${bboxArray}::double precision[],
            ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify({
              type,
              coordinates,
            })}), 4326),
            ${JSON.stringify(otherProperties)}::jsonb,
            NOW(),
            NOW()
          ) RETURNING *;
        `;
        })
        .filter(Boolean)
    );

    res.status(201).json({
      message: 'Roads created',
      count: createdRoads.length,
      roads: createdRoads,
    });
  } catch (error) {
    console.error('Error creating roads:', error);
    res
      .status(500)
      .json({ message: 'Failed to create road data', error: error.message });
  }
};

/**
 * Update a Road
 */
const updateRoad = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const { type, coordinates, bbox } = geometry;

    if (type !== 'LineString') {
      throw new Error('Invalid geometry type. Expected LineString.');
    }

    const bboxArray = bbox ? `{${bbox.join(',')}}` : 'NULL';

    const updatedRoad = await prisma.$executeRaw`
      UPDATE "Road"
      SET
        "name" = ${name},
        "description" = ${description},
        "geom" = ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
          geometry
        )}), 4326),
        "bbox" = ${bboxArray}::double precision[],
        "updatedAt" = NOW()
      WHERE "id" = ${Number(id)}
      RETURNING *;
    `;

    if (updatedRoad.length === 0) {
      return res.status(404).json({ message: 'Road not found' });
    }

    res.json(updatedRoad[0]);
  } catch (error) {
    console.error('Error updating road:', error);
    res.status(500).json({ message: 'Failed to update road data' });
  }
};

/**
 * Delete a Road
 */
const deleteRoad = async (req, res) => {
  const { id } = req.params;
  try {
    const deletedRoad = await prisma.road.delete({ where: { id: Number(id) } });
    res.json({ message: 'Road deleted', deletedRoad });
  } catch (error) {
    console.error('Error deleting road:', error);
    res.status(500).json({ message: 'Failed to delete road' });
  }
};

/**
 * Delete all Roads
 */
const deleteAllRoads = async (req, res) => {
  try {
    const deletedCount = await prisma.road.deleteMany();
    res.status(200).json({
      success: true,
      message: `Deleted ${deletedCount.count} roads`,
      count: deletedCount.count,
    });
  } catch (error) {
    console.error('Error deleting all roads:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete all roads',
      error: error.message,
    });
  }
};

module.exports = {
  getRoads,
  getRoad,
  createRoad,
  updateRoad,
  deleteAllRoads,
  deleteRoad,
};
