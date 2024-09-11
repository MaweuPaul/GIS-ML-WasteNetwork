const prisma = require('../lib/prisma');

export const getRivers = async (req, res) => {
  try {
    const rivers = await prisma.river.findMany();
    res.json(rivers);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch rivers' });
  }
};

export const getRiver = async (req, res) => {
  const { id } = req.params;
  try {
    const river = await prisma.river.findUnique({ where: { id: Number(id) } });
    res.json(river);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch river' });
  }
};

export const createRiver = async (req, res) => {
  const { name, length, geometry } = req.body;
  try {
    const newRiver = await prisma.river.create({
      data: { name, length, geometry },
    });
    res.status(200).json({ message: 'River data added', newRiver });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to create river data' });
  }
};

export const updateRiver = async (req, res) => {
  const { id } = req.params;
  const { name, length, geometry } = req.body;
  try {
    const updatedRiver = await prisma.river.update({
      where: { id: Number(id) },
      data: { name, length, geometry },
    });
    res.json(updatedRiver);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update river data' });
  }
};

export const deleteRiver = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.river.delete({ where: { id: Number(id) } });
    res.json({ message: 'River deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete river' });
  }
};
