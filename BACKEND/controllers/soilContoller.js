const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
const fs = require('fs');
const path = require('path');

dotenv.config();

const getSoils = async (req, res) => {
  try {
    const soils = await prisma.soil.findMany();
    res.json(soils);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch soils' });
  }
};

const getSoil = async (req, res) => {
  const { id } = req.params;
  try {
    const soil = await prisma.soil.findUnique({ where: { id: Number(id) } });
    res.json(soil);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch soil' });
  }
};

const createSoil = async (req, res) => {
  const { name, description } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const soil = await prisma.soil.create({
      data: {
        name,
        description,
        fileName: file.originalname,
        filePath: file.path,
        fileSize: file.size,
        fileType: file.mimetype,
      },
    });
    res.status(201).json({ message: 'File uploaded successfully', soil });
  } catch (error) {
    res.status(500).json({ message: 'Failed to upload file', error });
  }
};

const updateSoil = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedSoil = await prisma.soil.update({
      where: { id: Number(id) },
      data: { name, description, geometry },
    });
    res.json(updatedSoil);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update soil data' });
  }
};

const deleteSoil = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.soil.delete({ where: { id: Number(id) } });
    res.json({ message: 'Soil deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete soil' });
  }
};

module.exports = {
  getSoils,
  getSoil,
  createSoil,
  updateSoil,
  deleteSoil,
};
