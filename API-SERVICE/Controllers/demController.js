const prisma = require('../Lib/prisma.js');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
const { count } = require('console');
dotenv.config();

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

const getDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  try {
    const digitalElevationModel = await prisma.digitalElevationModel.findUnique(
      { where: { id: Number(id) } }
    );
    res.json(digitalElevationModel);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to fetch digital elevation model' });
  }
};

const createDigitalElevationModel = async (req, res) => {
  const { features } = req.body;

  if (!Array.isArray(features) || features.length === 0) {
    return res.status(400).json({
      message: 'Invalid input: Expected a non-empty array of features',
    });
  }

  try {
    const createdModels = await prisma.$transaction(
      features.map((feature) =>
        prisma.digitalElevationModel.create({
          data: {
            name: feature.name,
            bbox: feature.bbox,
            coordinates: feature.coordinates, // Prisma will automatically handle JSON serialization
            geometryType: feature.geometryType,
            elevation: feature.elevation,
          },
        })
      )
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

const updateDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedDigitalElevationModel =
      await prisma.digitalElevationModel.update({
        where: { id: Number(id) },
        data: {
          name,
          description,
          geom: JSON.parse(geometry),
        },
      });
    res.json(updatedDigitalElevationModel);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to update digital elevation model data' });
  }
};

const deleteDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.digitalElevationModel.delete({ where: { id: Number(id) } });
    res.json({ message: 'Digital elevation model deleted' });
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to delete digital elevation model' });
  }
};

const deleteAllDigitalElevationModel = async (req, res) => {
  try {
    const result = await prisma.digitalElevationModel.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Digital elevation cleared successfully',
      count: result.count,
    });
  } catch (error) {
    res.status(400).json({
      success: false,
      message: 'Digital elevation could not be cleared',
      error: error.message,
    });
  }
};

module.exports = {
  getDigitalElevationModels,
  getDigitalElevationModel,
  createDigitalElevationModel,
  updateDigitalElevationModel,
  deleteAllDigitalElevationModel,
  deleteDigitalElevationModel,
};
