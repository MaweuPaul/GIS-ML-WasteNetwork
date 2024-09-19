const prisma = require('../Lib/prisma.js');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

const getAreasOfInterest = async (req, res) => {
  try {
    const areasOfInterest = await prisma.areaOfInterest.findMany();
    res.json(areasOfInterest);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch areas of interest' });
  }
};

const getAreaOfInterest = async (req, res) => {
  const { id } = req.params;
  try {
    const areaOfInterest = await prisma.areaOfInterest.findUnique({
      where: { id: Number(id) },
    });
    res.json(areaOfInterest);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch area of interest' });
  }
};

const createAreaOfInterest = async (req, res) => {
  const { name, description, geometry } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const areaOfInterest = await prisma.areaOfInterest.create({
      data: {
        name,
        description,
        fileName: file.filename,
        filePath: file.path,
        fileSize: file.size,
        fileType: file.mimetype,
        geom: JSON.parse(geometry),
      },
    });
    res
      .status(201)
      .json({ message: 'File uploaded successfully', areaOfInterest });
  } catch (error) {
    console.error('Error in createAreaOfInterest:', error);
    res
      .status(500)
      .json({ message: 'Failed to upload file', error: error.message });
  }
};

const updateAreaOfInterest = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedAreaOfInterest = await prisma.areaOfInterest.update({
      where: { id: Number(id) },
      data: {
        name,
        description,
        geom: JSON.parse(geometry),
      },
    });
    res.json(updatedAreaOfInterest);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update area of interest data' });
  }
};

const deleteAreaOfInterest = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.areaOfInterest.delete({ where: { id: Number(id) } });
    res.json({ message: 'Area of interest deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete area of interest' });
  }
};

module.exports = {
  getAreasOfInterest,
  getAreaOfInterest,
  createAreaOfInterest,
  updateAreaOfInterest,
  deleteAreaOfInterest,
};
