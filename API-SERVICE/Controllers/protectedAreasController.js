const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Protected Areas
 */
const getProtectedAreas = async (req, res) => {
  try {
    const protectedAreas = await prisma.protectedArea.findMany();
    res.json(protectedAreas);
  } catch (error) {
    console.error('Error fetching protected areas:', error);
    res.status(500).json({ message: 'Failed to fetch protected areas' });
  }
};

/**
 * Fetch a single Protected Area by ID
 */
const getProtectedArea = async (req, res) => {
  const { id } = req.params;
  try {
    const protectedArea = await prisma.protectedArea.findUnique({
      where: { id: Number(id) },
    });
    if (!protectedArea) {
      return res.status(404).json({ message: 'Protected area not found' });
    }
    res.json(protectedArea);
  } catch (error) {
    console.error('Error fetching protected area:', error);
    res.status(500).json({ message: 'Failed to fetch protected area' });
  }
};

/**
 * Create multiple Protected Areas from GeoJSON
 */
const createProtectedArea = async (req, res) => {
  const { geojson } = req.body;

  if (!geojson || !geojson.features || !Array.isArray(geojson.features)) {
    return res.status(400).json({ message: 'Invalid GeoJSON data' });
  }

  try {
    const createdAreas = await prisma.$transaction(
      geojson.features.map((feature) => {
        const { type, coordinates, bbox } = feature.geometry;
        const { AREANAME, ...otherProperties } = feature.properties;

        const bboxArray = bbox ? `{${bbox.join(',')}}` : 'NULL';

        return prisma.$executeRaw`
          INSERT INTO "ProtectedArea" (
            "name",
            "type",
            "coordinates",
            "bbox",
            "geom",
            "properties",
            "createdAt",
            "updatedAt"
          ) VALUES (
            ${AREANAME},
            ${type},
            ${JSON.stringify(coordinates)}::jsonb,
            ${bboxArray}::double precision[],
            ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
              feature.geometry
            )}), 4326),
            ${JSON.stringify(otherProperties)}::jsonb,
            NOW(),
            NOW()
          ) RETURNING *;
        `;
      })
    );

    res.status(201).json({
      message: 'Protected areas created',
      count: createdAreas.length,
      areas: createdAreas,
    });
  } catch (error) {
    console.error('Error creating protected areas:', error);
    res.status(500).json({
      message: 'Failed to create protected area data',
      error: error.message,
    });
  }
};

/**
 * Update a Protected Area
 */
const updateProtectedArea = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const bboxArray = geometry.bbox ? `{${geometry.bbox.join(',')}}` : 'NULL';

    const updatedProtectedArea = await prisma.$executeRaw`
      UPDATE "ProtectedArea"
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

    if (updatedProtectedArea.length === 0) {
      return res.status(404).json({ message: 'Protected area not found' });
    }

    res.json(updatedProtectedArea[0]);
  } catch (error) {
    console.error('Error updating protected area:', error);
    res.status(500).json({ message: 'Failed to update protected area data' });
  }
};

/**
 * Delete a Protected Area
 */
const deleteProtectedArea = async (req, res) => {
  const { id } = req.params;
  try {
    const deletedArea = await prisma.protectedArea.delete({
      where: { id: Number(id) },
    });
    res.json({ message: 'Protected area deleted', deletedArea });
  } catch (error) {
    console.error('Error deleting protected area:', error);
    res.status(500).json({ message: 'Failed to delete protected area' });
  }
};

/**
 * Delete all Protected Areas
 */
const deleteAllProtectedAreas = async (req, res) => {
  try {
    const deletedCount = await prisma.protectedArea.deleteMany();
    res.status(200).json({
      success: true,
      message: `Deleted ${deletedCount.count} protected areas`,
      count: deletedCount.count,
    });
  } catch (error) {
    console.error('Error deleting all protected areas:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete all protected areas',
      error: error.message,
    });
  }
};

module.exports = {
  getProtectedAreas,
  getProtectedArea,
  createProtectedArea,
  updateProtectedArea,
  deleteProtectedArea,
  deleteAllProtectedAreas,
};
