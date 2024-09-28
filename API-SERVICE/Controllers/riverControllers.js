const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Rivers
 */
const getRivers = async (req, res) => {
  try {
    const rivers = await prisma.river.findMany();
    res.json(rivers);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch rivers' });
  }
};

/**
 * Fetch a single River by ID
 */
const getRiver = async (req, res) => {
  const { id } = req.params;
  try {
    const river = await prisma.river.findUnique({ where: { id: Number(id) } });
    if (!river) {
      return res.status(404).json({ message: 'River not found' });
    }
    res.json(river);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch river' });
  }
};

/**
 * Create multiple Rivers
 */
const createRiver = async (req, res) => {
  const { name, geojson } = req.body;

  try {
    // Validate input
    if (
      !name ||
      !geojson ||
      !geojson.features ||
      !Array.isArray(geojson.features)
    ) {
      return res.status(400).json({ message: 'Invalid input data' });
    }

    const createdRivers = await prisma.$transaction(
      geojson.features.map((feature) => {
        const { type, coordinates, bbox } = feature.geometry;
        const riverData = {
          name: name,
          type: type,
          coordinates: coordinates,
          bbox: bbox ? `{${bbox.join(',')}}` : 'NULL',
          properties: feature.properties,
        };

        // Insert using raw SQL to handle geom
        return prisma.$executeRaw`
          INSERT INTO "River" (
            "name",
            "type",
            "coordinates",
            "bbox",
            "properties",
            "geom",
            "createdAt",
            "updatedAt"
          ) VALUES (
            ${riverData.name},
            ${riverData.type},
            ${JSON.stringify(riverData.coordinates)}::jsonb,
            ${riverData.bbox}::double precision[],
            ${JSON.stringify(riverData.properties)}::jsonb,
            ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
              feature.geometry
            )}), 4326),
            NOW(),
            NOW()
          ) RETURNING *;
        `;
      })
    );

    res.status(201).json({ message: 'Rivers created', createdRivers });
  } catch (error) {
    console.error('Error creating river data:', error);
    res
      .status(500)
      .json({ message: 'Failed to create river data', error: error.message });
  }
};

/**
 * Update a River by ID
 */
const updateRiver = async (req, res) => {
  const { id } = req.params;
  const { name, description, geojson } = req.body;
  try {
    const feature = geojson.features[0];
    const { geometry, properties } = feature;

    const bboxArray = geometry.bbox ? `{${geometry.bbox.join(',')}}` : 'NULL';

    // Update using raw SQL
    const updatedRiver = await prisma.$executeRaw`
      UPDATE "River"
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

    if (updatedRiver.length === 0) {
      return res.status(404).json({ message: 'River not found' });
    }

    res.json(updatedRiver[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update river data' });
  }
};

/**
 * Delete a River by ID
 */
const deleteRiver = async (req, res) => {
  const { id } = req.params;
  try {
    const deleted = await prisma.river.delete({ where: { id: Number(id) } });
    res.json({ message: 'River deleted', deleted });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete river' });
  }
};

/**
 * Delete all Rivers
 */
const deleteAllRivers = async (req, res) => {
  try {
    const deletedCount = await prisma.river.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Rivers cleared successfully',
      count: deletedCount.count,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: 'Failed to clear rivers',
      error: error.message,
    });
  }
};

module.exports = {
  getRivers,
  getRiver,
  createRiver,
  updateRiver,
  deleteRiver,
  deleteAllRivers,
};
