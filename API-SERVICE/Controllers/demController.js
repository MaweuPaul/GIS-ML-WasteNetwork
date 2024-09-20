const prisma = require('../Lib/prisma.js');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');
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
  const { name, description, geometry } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No file uploaded' });
  }

  try {
    const digitalElevationModel = await prisma.digitalElevationModel.create({
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
      .json({ message: 'File uploaded successfully', digitalElevationModel });
  } catch (error) {
    console.error('Error in createDigitalElevationModel:', error);
    res
      .status(500)
      .json({ message: 'Failed to upload file', error: error.message });
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

module.exports = {
  getDigitalElevationModels,
  getDigitalElevationModel,
  createDigitalElevationModel,
  updateDigitalElevationModel,
  deleteDigitalElevationModel,
};
