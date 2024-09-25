const prisma = require('../Lib/prisma.js');

const fetchData = async (page = 1, limit = 100) => {
  const offset = (page - 1) * limit;
  try {
    const areasOfInterest = await prisma.areaOfInterest.findMany({
      skip: offset,
      take: limit,
    });
    const soils = await prisma.soil.findMany({ skip: offset, take: limit });
    const geologies = await prisma.geology.findMany({
      skip: offset,
      take: limit,
    });
    const digitalElevationModels = await prisma.digitalElevationModel.findMany({
      skip: offset,
      take: limit,
    });
    const protectedAreas = await prisma.protectedArea.findMany({
      skip: offset,
      take: limit,
    });
    const rivers = await prisma.river.findMany({ skip: offset, take: limit });
    const roads = await prisma.road.findMany({ skip: offset, take: limit });

    return {
      areasOfInterest,
      soils,
      geologies,
      digitalElevationModels,
      protectedAreas,
      rivers,
      roads,
    };
  } catch (error) {
    console.error('Error fetching data:', error);
    throw new Error('Failed to fetch data');
  }
};

module.exports = {
  fetchData,
};
