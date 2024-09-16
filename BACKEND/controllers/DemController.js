const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
const fs = require('fs');
const path = require('path');

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

const createDem = async (req, res) => {
  const { name, description } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const dem = await prisma.digitalElevationModel.create({
      data: {
        name,
        description,
        fileName: file.originalname,
        filePath: file.path,
        fileSize: file.size,
        fileType: file.mimetype,
      },
    });
    res.status(201).json({ message: 'File uploaded successfully', dem });
  } catch (error) {
    res.status(500).json({ message: 'Failed to upload file', error });
  }
};

const updateDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  const { name, resolution, geometry } = req.body;
  try {
    const updatedDigitalElevationModel =
      await prisma.digitalElevationModel.update({
        where: { id: Number(id) },
        data: { name, resolution, geometry },
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

module.exports = {
  getDigitalElevationModels,
  getDigitalElevationModel,
  createDem,
  updateDigitalElevationModel,
  deleteDigitalElevationModel,
};
