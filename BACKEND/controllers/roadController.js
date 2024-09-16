const prisma = require('../Lib/prisma.js');
const fs = require('fs');
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
  const { name, description } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const road = await prisma.road.create({
      data: {
        name,
        description,
        fileName: file.originalname,
        filePath: file.path,
        fileSize: file.size,
        fileType: file.mimetype,
      },
    });
    res.status(201).json({ message: 'File uploaded successfully', road });
  } catch (error) {
    res.status(500).json({ message: 'Failed to upload file', error });
  }
};

const updateRoad = async (req, res) => {
  const { id } = req.params;
  const { name, type, length, geometry } = req.body;
  try {
    const updatedRoad = await prisma.road.update({
      where: { id: Number(id) },
      data: { name, type, length, geometry },
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

module.exports = {
  getRoads,
  getRoad,
  createRoad,
  updateRoad,
  deleteRoad,
};
