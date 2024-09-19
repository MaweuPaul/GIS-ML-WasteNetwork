const prisma = require('../Lib/prisma.js');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

const getGeologies = async (req, res) => {
  try {
    const geologies = await prisma.geology.findMany();
    res.json(geologies);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch geologies' });
  }
};

const getGeology = async (req, res) => {
  const { id } = req.params;
  try {
    const geology = await prisma.geology.findUnique({
      where: { id: Number(id) },
    });
    res.json(geology);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch geology' });
  }
};

const createGeology = async (req, res) => {
  const { name, description, geometry } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const geology = await prisma.geology.create({
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
    res.status(201).json({ message: 'File uploaded successfully', geology });
  } catch (error) {
    console.error('Error in createGeology:', error);
    res
      .status(500)
      .json({ message: 'Failed to upload file', error: error.message });
  }
};

const updateGeology = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedGeology = await prisma.geology.update({
      where: { id: Number(id) },
      data: {
        name,
        description,
        geom: JSON.parse(geometry),
      },
    });
    res.json(updatedGeology);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update geology data' });
  }
};

const deleteGeology = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.geology.delete({ where: { id: Number(id) } });
    res.json({ message: 'Geology deleted' });
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
