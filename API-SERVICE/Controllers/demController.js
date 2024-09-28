const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Digital Elevation Models
 */
const getDigitalElevationModels = async (req, res) => {
  try {
    const digitalElevationModels =
      await prisma.digitalElevationModel.findMany();
    res.json(digitalElevationModels);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to fetch digital elevation models' });
  }
};

/**
 * Fetch a single Digital Elevation Model by ID
 */
const getDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  try {
    const digitalElevationModel = await prisma.digitalElevationModel.findUnique(
      {
        where: { id: Number(id) },
      }
    );
    if (!digitalElevationModel) {
      return res
        .status(404)
        .json({ message: 'Digital elevation model not found' });
    }
    res.json(digitalElevationModel);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to fetch digital elevation model' });
  }
};

/**
 * Create multiple Digital Elevation Models
 */
const createDigitalElevationModel = async (req, res) => {
  const { features } = req.body;

  if (!Array.isArray(features) || features.length === 0) {
    return res.status(400).json({
      message: 'Invalid input: Expected a non-empty array of features',
    });
  }

  try {
    const createdModels = await prisma.$transaction(
      features.map((feature) => {
        const { name, bbox, geometryType, coordinates, elevation } = feature;
        let { geometry } = feature;

        // Ensure geometry is not null
        if (!geometry) {
          throw new Error('Geometry is required');
        }

        // Use Prisma's executeRaw to handle geom field with PostGIS
        return prisma.$executeRaw`
          INSERT INTO "DigitalElevationModel" (
            "name",
            "bbox",
            "coordinates",
            "geometryType",
            "elevation",
            "geom",
            "createdAt",
            "updatedAt"
          ) VALUES (
            ${name},
            ${bbox ? `{${bbox.join(',')}}` : null}::double precision[],
            ${JSON.stringify(coordinates)}::jsonb,
            ${geometry.type},
            ${elevation},
            ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(geometry)}), 4326),
            NOW(),
            NOW()
          ) RETURNING *;
        `;
      })
    );

    res.status(201).json({
      message: 'Digital elevation models created',
      count: createdModels.length,
      models: createdModels,
    });
  } catch (error) {
    console.error('Error creating digital elevation models:', error);
    res.status(500).json({
      message: 'Failed to create digital elevation models',
      error: error.message,
    });
  }
};

/**
 * Update a Digital Elevation Model by ID
 */
const updateDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedDigitalElevationModel = await prisma.$executeRaw`
      UPDATE "DigitalElevationModel"
      SET 
        "name" = ${name},
        "description" = ${description},
        "geom" = ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
          geometry
        )}), 4326),
        "updatedAt" = NOW()
      WHERE id = ${Number(id)}
      RETURNING *;
    `;

    if (updatedDigitalElevationModel.length === 0) {
      return res
        .status(404)
        .json({ message: 'Digital elevation model not found' });
    }

    res.json(updatedDigitalElevationModel[0]);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to update digital elevation model data' });
  }
};

/**
 * Delete a Digital Elevation Model by ID
 */
const deleteDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  try {
    const deleted = await prisma.digitalElevationModel.delete({
      where: { id: Number(id) },
    });
    res.json({ message: 'Digital elevation model deleted', deleted });
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to delete digital elevation model' });
  }
};

/**
 * Delete all Digital Elevation Models
 */
const deleteAllDigitalElevationModels = async (req, res) => {
  try {
    const deletedCount = await prisma.digitalElevationModel.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Digital elevation models cleared successfully',
      count: deletedCount.count,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: 'Failed to clear digital elevation models',
      error: error.message,
    });
  }
};

module.exports = {
  getDigitalElevationModels,
  getDigitalElevationModel,
  createDigitalElevationModel,
  updateDigitalElevationModel,
  deleteDigitalElevationModel,
  deleteAllDigitalElevationModels,
};
