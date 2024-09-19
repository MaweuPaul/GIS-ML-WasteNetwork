const prisma = require('../Lib/prisma.js');
const dotenv = require('dotenv');

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

const createSoils = async (req, res) => {
  const { name, type, features } = req.body;
  console.log(features);

  if (type !== 'FeatureCollection' || !Array.isArray(features)) {
    return res
      .status(400)
      .json({ message: 'Invalid input: Expected a FeatureCollection' });
  }

  try {
    const createdSoils = await prisma.$transaction(
      features.map((feature) =>
        prisma.soil.create({
          data: {
            type: feature.type,
            name: name,
            geometryType: feature.geometry.type,
            coordinates: feature.geometry.coordinates,
            bbox: feature.geometry.bbox,
            area: feature.properties.AREA,
            perimeter: feature.properties.PERIMETER,
            kensots: feature.properties.KENSOTS_,
            kensotsId: feature.properties.KENSOTS_ID,
            suid: feature.properties.SUID,
            newSuid: feature.properties.NEWSUID,
            slop: feature.properties.SLOP,
            reli: feature.properties.RELI,
            lndf: feature.properties.LNDF,
            hyps: feature.properties.HYPS,
            lith: feature.properties.LITH,
            isocsuid: feature.properties.ISOCSUID,
            suid12: feature.properties.SUID_12,
            firstOfCla: feature.properties.FIRSTOFCLA,
            firstOfSum: feature.properties.FIRSTOFSUM,
            link: feature.properties.LINK,
          },
        })
      )
    );

    res.status(201).json({
      message: 'Soil data created',
      count: createdSoils.length,
      soils: createdSoils,
    });
  } catch (error) {
    console.error('Error creating soil data:', error);
    res
      .status(500)
      .json({ message: 'Failed to create soil data', error: error.message });
  }
};

const updateSoil = async (req, res) => {
  const { id } = req.params;
  const { name, description, geojson } = req.body;
  try {
    let parsedGeojson;
    if (typeof geojson === 'string') {
      parsedGeojson = JSON.parse(geojson);
    } else {
      parsedGeojson = geojson;
    }

    if (
      parsedGeojson.type !== 'FeatureCollection' ||
      !Array.isArray(parsedGeojson.features)
    ) {
      return res.status(400).json({ message: 'Invalid GeoJSON format' });
    }

    const updatedSoil = await prisma.soil.update({
      where: { id: Number(id) },
      data: {
        name,
        description,
        geom: parsedGeojson.features[0].geometry,
        properties: parsedGeojson.features[0].properties,
      },
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
const deleteAllSoils = async (req, res) => {
  try {
    const deletedCount = await prisma.soil.deleteMany();
    res.status(200).json({
      message: `Deleted ${deletedCount.count} soil records`,
      count: deletedCount.count,
    });
  } catch (error) {
    console.error('Error deleting all soils:', error);
    res
      .status(500)
      .json({ message: 'Failed to delete all soils', error: error.message });
  }
};

module.exports = {
  getSoils,
  getSoil,
  createSoils,
  updateSoil,
  deleteSoil,
  deleteAllSoils,
};
