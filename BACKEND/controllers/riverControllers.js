const prisma = require('../Lib/prisma.js');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

const getRivers = async (req, res) => {
  try {
    const rivers = await prisma.river.findMany();
    res.json(rivers);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch rivers' });
  }
};

const getRiver = async (req, res) => {
  const { id } = req.params;
  try {
    const river = await prisma.river.findUnique({ where: { id: Number(id) } });
    res.json(river);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch river' });
  }
};

const createRiver = async (req, res) => {
  const { name, description } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const river = await prisma.river.create({
      data: {
        name,
        description,
        fileName: file.originalname,
        filePath: file.path,
        fileSize: file.size,
        fileType: file.mimetype,
      },
    });
    res.status(201).json({ message: 'File uploaded successfully', river });
  } catch (error) {
    res.status(500).json({ message: 'Failed to upload file', error });
  }
};

const updateRiver = async (req, res) => {
  const { id } = req.params;
  const { name, length, geometry } = req.body;
  try {
    const updatedRiver = await prisma.river.update({
      where: { id: Number(id) },
      data: { name, length, geometry },
    });
    res.json(updatedRiver);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update river data' });
  }
};

const deleteRiver = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.river.delete({ where: { id: Number(id) } });
    res.json({ message: 'River deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete river' });
  }
};

module.exports = {
  getRivers,
  getRiver,
  createRiver,
  updateRiver,
  deleteRiver,
};
