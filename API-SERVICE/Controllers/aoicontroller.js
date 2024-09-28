const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');
dotenv.config();

/**
 * Fetch all Areas of Interest
 */
const getAreasOfInterest = async (req, res) => {
  try {
    const areasOfInterest = await prisma.areaOfInterest.findMany();
    res.json(areasOfInterest);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch areas of interest' });
  }
};

/**
 * Fetch a single Area of Interest by ID
 */
const getAreaOfInterest = async (req, res) => {
  const { id } = req.params;
  try {
    const areaOfInterest = await prisma.areaOfInterest.findUnique({
      where: { id: Number(id) },
    });
    if (!areaOfInterest) {
      return res.status(404).json({ message: 'Area of interest not found' });
    }
    res.json(areaOfInterest);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch area of interest' });
  }
};

/**
 * Create an Area of Interest
 */
const createAreaOfInterest = async (req, res) => {
  const { feature } = req.body;

  if (!feature || feature.type !== 'Feature' || !feature.geometry) {
    return res.status(400).json({ message: 'Invalid GeoJSON feature' });
  }

  try {
    const {
      type,
      geometry,
      properties: {
        OBJECTID,
        ISO,
        NAME_0,
        ID_1,
        NAME_1,
        ID_2,
        NAME_2,
        CCN_2,
        TYPE_2,
        ENGTYPE_2,
        Shape_Leng,
        Shape_Area,
        Area,
      },
    } = feature;

    const bboxArray = geometry.bbox ? `{${geometry.bbox.join(',')}}` : 'NULL';

    const areaOfInterest = await prisma.$executeRaw`
      INSERT INTO "AreaOfInterest" (
        "type",
        "name",
        "geometryType",
        "coordinates",
        "bbox",
        "objectId",
        "iso",
        "country",
        "provinceId",
        "provinceName",
        "districtId",
        "districtName",
        "ccn",
        "type2",
        "engType",
        "shapeLeng",
        "shapeArea",
        "area",
        "properties",
        "geom",
        "createdAt",
        "updatedAt"
      ) VALUES (
        ${type},
        ${NAME_2},
        ${geometry.type},
        ${JSON.stringify(geometry.coordinates)}::jsonb,
        ${bboxArray}::double precision[],
        ${OBJECTID},
        ${ISO},
        ${NAME_0},
        ${ID_1},
        ${NAME_1},
        ${ID_2},
        ${NAME_2},
        ${CCN_2},
        ${TYPE_2},
        ${ENGTYPE_2},
        ${Shape_Leng},
        ${Shape_Area},
        ${Area},
        ${JSON.stringify(feature.properties)}::jsonb,
        ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(geometry)}), 4326),
        NOW(),
        NOW()
      ) RETURNING *;
    `;

    res.status(201).json({
      message: 'Area of interest created successfully',
      areaOfInterest: areaOfInterest[0],
    });
  } catch (error) {
    console.error('Error in createAreaOfInterest:', error);
    res.status(500).json({
      message: 'Failed to create area of interest',
      error: error.message,
    });
  }
};

/**
 * Update an Area of Interest by ID
 */
const updateAreaOfInterest = async (req, res) => {
  const { id } = req.params;
  const { feature } = req.body;

  if (!feature || feature.type !== 'Feature' || !feature.geometry) {
    return res.status(400).json({ message: 'Invalid GeoJSON feature' });
  }

  try {
    const {
      type,
      geometryType,
      geometry,
      coordinates,
      bbox,
      properties: {
        OBJECTID,
        ISO,
        NAME_0,
        ID_1,
        NAME_1,
        ID_2,
        NAME_2,
        CCN_2,
        TYPE_2,
        ENGTYPE_2,
        Shape_Leng,
        Shape_Area,
        Area,
      },
    } = feature;

    const updatedAreaOfInterest = await prisma.$executeRaw`
      UPDATE "AreaOfInterest"
      SET
        "type" = ${type},
        "name" = ${NAME_2},
        "geometryType" = ${geometry.type},
        "coordinates" = ${coordinates},
        "bbox" = ${bbox},
        "objectId" = ${OBJECTID},
        "iso" = ${ISO},
        "country" = ${NAME_0},
        "provinceId" = ${ID_1},
        "provinceName" = ${NAME_1},
        "districtId" = ${ID_2},
        "districtName" = ${NAME_2},
        "ccn" = ${CCN_2},
        "type2" = ${TYPE_2},
        "engType" = ${ENGTYPE_2},
        "shapeLeng" = ${Shape_Leng},
        "shapeArea" = ${Shape_Area},
        "area" = ${Area},
        "properties" = ${JSON.stringify(feature.properties)},
        "geom" = ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
          geometry
        )}), 4326),
        "updatedAt" = NOW()
      WHERE "id" = ${Number(id)}
      RETURNING *;
    `;

    if (updatedAreaOfInterest.length === 0) {
      return res.status(404).json({ message: 'Area of interest not found' });
    }

    res.json(updatedAreaOfInterest[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update area of interest data' });
  }
};

/**
 * Delete an Area of Interest by ID
 */
const deleteAreaOfInterest = async (req, res) => {
  const { id } = req.params;
  try {
    const deleted = await prisma.areaOfInterest.delete({
      where: { id: Number(id) },
    });
    res.json({ message: 'Area of interest deleted', deleted });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete area of interest' });
  }
};

/**
 * Delete all Areas of Interest
 */
const deleteAllAreasOfInterest = async (req, res) => {
  try {
    const deletedCount = await prisma.areaOfInterest.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Areas of interest cleared successfully',
      count: deletedCount.count,
    });
  } catch (error) {
    console.error('Error deleting all areas of interest:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to clear areas of interest',
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
