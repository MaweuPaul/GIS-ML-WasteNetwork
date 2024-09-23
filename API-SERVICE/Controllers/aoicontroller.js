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
  const { feature } = req.body;

  try {
    const areaOfInterest = await prisma.areaOfInterest.create({
      data: {
        type: feature.type,
        name: feature.properties.NAME_2,
        geometryType: feature.geometry.type,
        coordinates: feature.geometry.coordinates,
        bbox: feature.geometry.bbox,
        objectId: feature.properties.OBJECTID,
        iso: feature.properties.ISO,
        country: feature.properties.NAME_0,
        provinceId: feature.properties.ID_1,
        provinceName: feature.properties.NAME_1,
        districtId: feature.properties.ID_2,
        districtName: feature.properties.NAME_2,
        ccn: feature.properties.CCN_2,
        type2: feature.properties.TYPE_2,
        engType: feature.properties.ENGTYPE_2,
        shapeLeng: feature.properties.Shape_Leng,
        shapeArea: feature.properties.Shape_Area,
        area: feature.properties.Area,
        properties: feature.properties,
      },
    });
    res.status(201).json({
      message: 'Area of interest created successfully',
      areaOfInterest,
    });
  } catch (error) {
    console.error('Error in createAreaOfInterest:', error);
    res.status(500).json({
      message: 'Failed to create area of interest',
      error: error.message,
    });
  }
};

const updateAreaOfInterest = async (req, res) => {
  const { id } = req.params;
  const { feature } = req.body;
  try {
    const updatedAreaOfInterest = await prisma.areaOfInterest.update({
      where: { id: Number(id) },
      data: {
        type: feature.type,
        name: feature.properties.NAME_2,
        geometryType: feature.geometry.type,
        coordinates: feature.geometry.coordinates,
        bbox: feature.geometry.bbox,
        objectId: feature.properties.OBJECTID,
        iso: feature.properties.ISO,
        country: feature.properties.NAME_0,
        provinceId: feature.properties.ID_1,
        provinceName: feature.properties.NAME_1,
        districtId: feature.properties.ID_2,
        districtName: feature.properties.NAME_2,
        ccn: feature.properties.CCN_2,
        type2: feature.properties.TYPE_2,
        engType: feature.properties.ENGTYPE_2,
        shapeLeng: feature.properties.Shape_Leng,
        shapeArea: feature.properties.Shape_Area,
        area: feature.properties.Area,
        properties: feature.properties,
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

const deleteAllAreasOfInterest = async (req, res) => {
  try {
    const countBefore = await prisma.areaOfInterest.count();
    console.log(`Areas of Interest count before delete: ${countBefore}`);

    const deletedCount = await prisma.areaOfInterest.deleteMany();

    const countAfter = await prisma.areaOfInterest.count();
    console.log(`Areas of Interest count after delete: ${countAfter}`);
    console.log(`Deleted ${deletedCount.count} Areas of Interest`);

    res.status(200).json({
      success: true,
      message: `Deleted ${deletedCount.count} areas of interest`,
      count: deletedCount.count,
    });
  } catch (error) {
    console.error('Error deleting all areas of interest:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete all areas of interest',
      error: error.message,
    });
  }
};

module.exports = {
  getAreasOfInterest,
  getAreaOfInterest,
  createAreaOfInterest,
  updateAreaOfInterest,
  deleteAreaOfInterest,
  deleteAllAreasOfInterest,
};
