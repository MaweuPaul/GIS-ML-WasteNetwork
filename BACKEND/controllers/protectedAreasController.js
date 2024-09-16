const prisma = require('../Lib/prisma.js');
const fs = require('fs');
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
  const { name, description } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const protectedArea = await prisma.protectedArea.create({
      data: {
        name,
        description,
        fileName: file.originalname,
        filePath: file.path,
        fileSize: file.size,
        fileType: file.mimetype,
      },
    });
    res
      .status(201)
      .json({ message: 'File uploaded successfully', protectedArea });
  } catch (error) {
    res.status(500).json({ message: 'Failed to upload file', error });
  }
};

const updateProtectedArea = async (req, res) => {
  const { id } = req.params;
  const { name, description, area, geometry } = req.body;
  try {
    const updatedProtectedArea = await prisma.protectedArea.update({
      where: { id: Number(id) },
      data: { name, description, area, geometry },
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

module.exports = {
  getProtectedAreas,
  getProtectedArea,
  createProtectedArea,
  updateProtectedArea,
  deleteProtectedArea,
};
