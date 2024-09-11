const prisma = require('../lib/prisma');

export const getRoads = async (req, res) => {
  try {
    const roads = await prisma.road.findMany();
    res.json(roads);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch roads' });
  }
};

export const getRoad = async (req, res) => {
  const { id } = req.params;
  try {
    const road = await prisma.road.findUnique({ where: { id: Number(id) } });
    res.json(road);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch road' });
  }
};

export const createRoad = async (req, res) => {
  const { name, type, length, geometry } = req.body;
  try {
    const newRoad = await prisma.road.create({
      data: { name, type, length, geometry },
    });
    res.status(200).json({ message: 'Road data added', newRoad });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to create road data' });
  }
};

export const updateRoad = async (req, res) => {
  const { id } = req.params;
  const { name, type, length, geometry } = req.body;
  try {
    const updatedRoad = await prisma.road.update({
      where: { id: Number(id) },
      data: { name, type, length, geometry },
    });
    res.json(updatedRoad);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update road data' });
  }
};

export const deleteRoad = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.road.delete({ where: { id: Number(id) } });
    res.json({ message: 'Road deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete road' });
  }
};
