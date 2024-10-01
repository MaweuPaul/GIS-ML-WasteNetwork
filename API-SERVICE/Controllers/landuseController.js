const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const uploadLandUseRaster = async (req, res) => {
  const { description } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ message: 'No TIFF file uploaded.' });
  }

  try {
    // Read the file buffer
    const rasterBuffer = file.buffer;

    // Insert into the database
    const landUseRaster = await prisma.landUseRaster.create({
      data: {
        description,
        raster: rasterBuffer,
      },
    });

    res.status(201).json({
      message: 'Land Use Raster uploaded successfully.',
      landUseRaster,
    });
  } catch (error) {
    console.error('Error uploading Land Use Raster:', error);
    res.status(500).json({
      message: 'Failed to upload Land Use Raster.',
      error: error.message,
    });
  }
};

const getLandUseRaster = async (req, res) => {
  const { id } = req.params;

  try {
    const raster = await prisma.landUseRaster.findUnique({
      where: { id: Number(id) },
    });

    if (!raster) {
      return res.status(404).json({ message: 'Land Use Raster not found.' });
    }

    res.setHeader('Content-Type', 'image/tiff');
    res.send(raster.raster);
  } catch (error) {
    console.error('Error fetching Land Use Raster:', error);
    res.status(500).json({
      message: 'Failed to fetch Land Use Raster.',
      error: error.message,
    });
  }
};

const deleteAllLanduse = async (req, res) => {
  try {
    const deletedCount = await prisma.landUseRaster.deleteMany();
    res.status(200).json({
      success: true,
      message: 'Landuse data deleted successfully',
      count: deletedCount.count,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to clear landuse ',
      error: error.message,
    });
  }
};
module.exports = { uploadLandUseRaster, getLandUseRaster, deleteAllLanduse };
