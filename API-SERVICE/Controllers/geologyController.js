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
  const { name, description, geometry } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const geology = await prisma.$executeRaw`
      INSERT INTO "Geology" (
        "name",
        "description",
        "fileName",
        "filePath",
        "fileSize",
        "fileType",
        "geom",
        "properties",
        "createdAt",
        "updatedAt"
      ) VALUES (
        ${name},
        ${description},
        ${file.filename},
        ${file.path},
        ${file.size},
        ${file.mimetype},
        ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(geometry)}), 4326),
        ${JSON.stringify(req.body.properties || {})},
        NOW(),
        NOW()
      ) RETURNING *;
    `;

    res.status(201).json({ message: 'File uploaded successfully', geology });
  } catch (error) {
    console.error('Error in createGeology:', error);
    res
      .status(500)
      .json({ message: 'Failed to upload file', error: error.message });
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

module.exports = {
  getGeologies,
  getGeology,
  createGeology,
  updateGeology,
  deleteGeology,
};
