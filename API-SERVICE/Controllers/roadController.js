const prisma = require('../Lib/prisma.js');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

const getRoads = async (req, res) => {
  try {
    const roads = await prisma.road.findMany();
    res.json(roads);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch roads' });
  }
};

const getRoad = async (req, res) => {
  const { id } = req.params;
  try {
    const road = await prisma.road.findUnique({ where: { id: Number(id) } });
    res.json(road);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch road' });
  }
};

const createRoad = async (req, res) => {
  const { geojson } = req.body;

  try {
    const createdRoads = await Promise.all(
      geojson.features.map(async (feature) => {
        const { type, coordinates } = feature.geometry;
        const { name, ...otherProperties } = feature.properties;

        return await prisma.road.create({
          data: {
            name: name || 'Unnamed Road',
            type: type,
            coordinates: coordinates,
            properties: otherProperties,
          },
        });
      })
    );

    res.status(201).json({ message: 'Roads created', createdRoads });
  } catch (error) {
    console.error('Error creating roads:', error);
    res
      .status(500)
      .json({ message: 'Failed to create road data', error: error.message });
  }
};

const updateRoad = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedRoad = await prisma.road.update({
      where: { id: Number(id) },
      data: {
        name,
        description,
        geom: JSON.parse(geometry),
      },
    });
    res.json(updatedRoad);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update road data' });
  }
};

const deleteRoad = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.road.delete({ where: { id: Number(id) } });
    res.json({ message: 'Road deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete road' });
  }
};

const deleteAllRoads = async (req, res) => {
  try {
    const deletedCount = await prisma.road.deleteMany();
    res.status(200).json({
      message: `Deleted ${deletedCount.count} roads`,
      count: deletedCount.count,
    });
  } catch (error) {
    console.error('Error deleting all roads:', error);
    res
      .status(500)
      .json({ message: 'Failed to delete all roads', error: error.message });
  }
};
module.exports = {
  getRoads,
  getRoad,
  createRoad,
  updateRoad,
  deleteRoad,
  deleteAllRoads,
};
