const prisma = require('../Lib/prisma.js');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

const getProtectedAreas = async (req, res) => {
  try {
    const protectedAreas = await prisma.protectedArea.findMany();
    res.json(protectedAreas);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch protected areas' });
  }
};

const getProtectedArea = async (req, res) => {
  const { id } = req.params;
  try {
    const protectedArea = await prisma.protectedArea.findUnique({
      where: { id: Number(id) },
    });
    res.json(protectedArea);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch protected area' });
  }
};

const createProtectedArea = async (req, res) => {
  const { name, geojson } = req.body;

  try {
    const createdAreas = await Promise.all(
      geojson.features.map(async (feature) => {
        const { type, coordinates, bbox } = feature.geometry;
        const { AREANAME, ...otherProperties } = feature.properties;

        return await prisma.protectedArea.create({
          data: {
            name: AREANAME,
            type: type,
            coordinates: coordinates,
            bbox: bbox || [],
            properties: otherProperties,
          },
        });
      })
    );

    res.status(201).json({ message: 'Protected areas created', createdAreas });
  } catch (error) {
    console.error('Error creating protected areas:', error);
    res.status(500).json({
      message: 'Failed to create protected area data',
      error: error.message,
    });
  }
};

const updateProtectedArea = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedProtectedArea = await prisma.protectedArea.update({
      where: { id: Number(id) },
      data: {
        name,
        description,
        geom: JSON.parse(geometry),
      },
    });
    res.json(updatedProtectedArea);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update protected area data' });
  }
};

const deleteProtectedArea = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.protectedArea.delete({ where: { id: Number(id) } });
    res.json({ message: 'Protected area deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete protected area' });
  }
};
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
