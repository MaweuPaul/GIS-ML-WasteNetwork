const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const checkDatabaseEmpty = async (req, res) => {
  try {
    const soilCount = await prisma.soil.count();
    const riverCount = await prisma.river.count();
    const protectedAreaCount = await prisma.protectedArea.count();
    const areaOfInterestCount = await prisma.areaOfInterest.count();
    const digitalElevationModelCount =
      await prisma.digitalElevationModel.count();
    const landUseRasterCount = await prisma.landUseRaster.count();
    const settlementCount = await prisma.settlement.count();

    const totalCount =
      soilCount +
      riverCount +
      protectedAreaCount +
      areaOfInterestCount +
      digitalElevationModelCount +
      landUseRasterCount +
      settlementCount;

    res.status(200).json({
      isEmpty: totalCount === 0,
      counts: {
        soils: soilCount,
        rivers: riverCount,
        protectedAreas: protectedAreaCount,
        areasOfInterest: areaOfInterestCount,
        digitalElevationModels: digitalElevationModelCount,
        landUseRasters: landUseRasterCount,
        settlements: settlementCount,
      },
    });
  } catch (error) {
    console.error('Error checking database:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to check database',
      error: error.message,
    });
  }
};

module.exports = { checkDatabaseEmpty };
