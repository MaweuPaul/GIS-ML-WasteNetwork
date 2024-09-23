const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');

dotenv.config();

const getRivers = async (req, res) => {
  try {
    const rivers = await prisma.river.findMany();
    res.json(rivers);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch rivers' });
  }
};

const getRiver = async (req, res) => {
  const { id } = req.params;
  try {
    const river = await prisma.river.findUnique({ where: { id: Number(id) } });
    res.json(river);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch river' });
  }
};
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

    const createdRivers = await Promise.all(
      geojson.features.map(async (feature) => {
        const { type, coordinates, bbox } = feature.geometry;
        const riverData = {
          name: name,
          type: type,
          coordinates: coordinates,
          bbox: bbox || [],
          properties: feature.properties,
        };

        return await prisma.river.create({
          data: riverData,
        });
      })
    );

    res.status(201).json({ message: 'Rivers created', createdRivers });
  } catch (error) {
    res
      .status(500)
      .json({ message: 'Failed to create river data', error: error.message });
  }
};
const updateRiver = async (req, res) => {
  const { id } = req.params;
  const { name, description, geojson } = req.body;
  try {
    const geom = geojson.features[0].geometry;
    const properties = geojson.features[0].properties;

    const updatedRiver = await prisma.river.update({
      where: { id: Number(id) },
      data: {
        name,
        description,
        geom,
        properties,
      },
    });
    res.json(updatedRiver);
  } catch (error) {
    res.status(500).json({ message: 'Failed to update river data' });
  }
};

const deleteRiver = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.river.delete({ where: { id: Number(id) } });
    res.json({ message: 'River deleted' });
  } catch (error) {
    res.status(500).json({ message: 'Failed to delete river' });
  }
};
const deleteAllRivers = async (req, res) => {
  try {
    const countBefore = await prisma.river.count();
    console.log(`Rivers count before delete: ${countBefore}`);

    const deletedCount = await prisma.river.deleteMany();

    const countAfter = await prisma.river.count();
    console.log(`Rivers count after delete: ${countAfter}`);

    res.status(200).json({
      success: true,
      message: `Deleted ${deletedCount.count} rivers`,
      count: deletedCount.count,
    });
  } catch (error) {
    console.error('Error deleting all rivers:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete all rivers',
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
