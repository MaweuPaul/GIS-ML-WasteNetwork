const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Geologies
 */
const getGeologies = async (req, res) => {
  try {
    const geologies = await prisma.geology.findMany();
    res.json(geologies);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch geologies' });
  }
};

/**
 * Fetch a single Geology by ID
 */
const getGeology = async (req, res) => {
  const { id } = req.params;
  try {
    const geology = await prisma.geology.findUnique({
      where: { id: Number(id) },
    });
    if (!geology) {
      return res.status(404).json({ message: 'Geology not found' });
    }
    res.json(geology);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch geology' });
  }
};

/**
 * Create a Geology with File Upload
 */
const createGeology = async (req, res) => {
  const { name, geometry, bbox, properties } = req.body;

  try {
    // Format the bbox array correctly for PostgreSQL
    const formattedBbox = bbox ? `{${bbox.join(',')}}` : null;

    // Convert the geometry to a string
    const geometryString = JSON.stringify(geometry);

    // Use text for the geometry instead of a parameter
    const [geology] = await prisma.$queryRaw`
      INSERT INTO "Geology" (
        "name",
        "geom",
        "bbox",
        "properties",
        "createdAt",
        "updatedAt"
      ) VALUES (
        ${name},
        ST_SetSRID(ST_GeomFromGeoJSON(${geometryString}::jsonb), 4326),
        ${formattedBbox}::float[],
        ${JSON.stringify(properties)}::jsonb,
        NOW(),
        NOW()
      ) RETURNING id, name, bbox, properties, "createdAt", "updatedAt"
    `;

    // Fetch the geometry separately
    const [geomResult] = await prisma.$queryRaw`
      SELECT ST_AsGeoJSON("geom") as geom
      FROM "Geology"
      WHERE id = ${geology.id}
    `;

    geology.geom = JSON.parse(geomResult.geom);

    res.status(201).json({ message: 'Geology created successfully', geology });
  } catch (error) {
    console.error('Error in createGeology:', error);
    res
      .status(500)
      .json({ message: 'Failed to create geology', error: error.message });
  }
};
/**
 * Update a Geology by ID
 */
const updateGeology = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedGeology = await prisma.$executeRaw`
      UPDATE "Geology"
      SET
        "name" = ${name},
        "description" = ${description},
        "geom" = ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
          geometry
        )}), 4326),
        "updatedAt" = NOW()
      WHERE "id" = ${Number(id)}
      RETURNING *;
    `;

    if (updatedGeology.length === 0) {
      return res.status(404).json({ message: 'Geology not found' });
    }

    res.json(updatedGeology[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update geology data' });
  }
};

/**
 * Delete a Geology by ID
 */
const deleteGeology = async (req, res) => {
  const { id } = req.params;
  try {
    const deleted = await prisma.geology.delete({ where: { id: Number(id) } });
    res.json({ message: 'Geology deleted', deleted });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete geology' });
  }
};

const deleteAllGeology = async (req, res) => {
  try {
    const deletedCount = await prisma.geology.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Geology cleared successfully',
      count: deletedCount.count,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: 'Failed to clear geology',
      error: error.message,
    });
  }
};
module.exports = {
  getGeologies,
  getGeology,
  createGeology,
  updateGeology,
  deleteGeology,
  deleteAllGeology,
};
