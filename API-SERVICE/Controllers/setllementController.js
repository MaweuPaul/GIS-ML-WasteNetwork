const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Settlements
 */
const getSettlements = async (req, res) => {
  try {
    const settlements = await prisma.settlement.findMany();
    res.json(settlements);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch settlements' });
  }
};

/**
 * Fetch a single Settlement by ID
 */
const getSettlement = async (req, res) => {
  const { id } = req.params;
  try {
    const settlement = await prisma.settlement.findUnique({
      where: { id: Number(id) },
    });
    if (!settlement) {
      return res.status(404).json({ message: 'Settlement not found' });
    }
    res.json(settlement);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch settlement' });
  }
};

/**
 * Create multiple Settlements
 */
const createSettlements = async (req, res) => {
  const features = Array.isArray(req.body) ? req.body : [req.body];

  if (features.length === 0) {
    return res.status(400).json({
      message: 'Invalid input: Expected at least one feature',
    });
  }

  try {
    const createdSettlements = await prisma.$transaction(
      features.map((feature) => {
        const { name, type, geometry, properties } = feature;
        console.log('geometry is ', geometry);

        // Ensure geometry is not null
        if (!geometry) {
          throw new Error('Geometry is required');
        }

        // Extract geometry details
        const { type: geometryType, coordinates } = geometry;

        // Use Prisma's executeRaw to handle geom field with PostGIS
        return prisma.$executeRaw`
              INSERT INTO "Settlement" (
                "name",
                "type",
                "geometryType",
                "coordinates",
                "properties",
                "geom",
                "createdAt",
                "updatedAt"
              ) VALUES (
                ${name},       
                ${type},
                ${geometryType},
                ${JSON.stringify(coordinates)}::jsonb,
                ${JSON.stringify(properties)}::jsonb,
                ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
                  geometry
                )}), 4326),
                NOW(),
                NOW()
              ) RETURNING *;
            `;
      })
    );

    res.status(201).json({
      message: 'Settlements created',
      count: createdSettlements.length,
      settlements: createdSettlements,
    });
  } catch (error) {
    console.error('Error creating settlements:', error);
    res.status(500).json({
      message: 'Failed to create settlements',
      error: error.message,
    });
  }
};

/**
 * Update a Settlement by ID
 */
const updateSettlement = async (req, res) => {
  const { id } = req.params;
  const { name, population, type, geometry } = req.body;
  try {
    const updatedSettlement = await prisma.$executeRaw`
      UPDATE "Settlement"
      SET 
        "name" = ${name},
        "population" = ${population},
        "type" = ${type},
        "geom" = ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
          geometry
        )}), 4326),
        "updatedAt" = NOW()
      WHERE id = ${Number(id)}
      RETURNING *;
    `;

    if (updatedSettlement.length === 0) {
      return res.status(404).json({ message: 'Settlement not found' });
    }

    res.json(updatedSettlement[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update settlement data' });
  }
};

/**
 * Delete a Settlement by ID
 */
const deleteSettlement = async (req, res) => {
  const { id } = req.params;
  try {
    const deleted = await prisma.settlement.delete({
      where: { id: Number(id) },
    });
    res.json({ message: 'Settlement deleted', deleted });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete settlement' });
  }
};

/**
 * Delete all Settlements
 */
const deleteAllSettlements = async (req, res) => {
  try {
    const deletedCount = await prisma.settlement.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Settlements cleared successfully',
      count: deletedCount.count,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: 'Failed to clear settlements',
      error: error.message,
    });
  }
};

module.exports = {
  getSettlements,
  getSettlement,
  createSettlements,
  updateSettlement,
  deleteSettlement,
  deleteAllSettlements,
};
