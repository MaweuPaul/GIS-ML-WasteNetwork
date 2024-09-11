const prisma = require('../lib/prisma');

export const getSoils = async (req, res) => {
  try {
    const soils = await prisma.soil.findMany();
    res.json(soils);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch soils' });
  }
};

export const getSoil = async (req, res) => {
  const { id } = req.params;
  try {
    const soil = await prisma.soil.findUnique({ where: { id: Number(id) } });
    res.json(soil);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch soil' });
  }
};

export const createSoil = async (req, res) => {
  const { name, description, geometry } = req.body;
  try {
    const newSoil = await prisma.soil.create({
      data: { name, description, geometry },
    });
    res.status(200).json({ message: 'Soil data added', newSoil });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to create soil data' });
  }
};

export const updateSoil = async (req, res) => {
  const { id } = req.params;
  const { name, description, geometry } = req.body;
  try {
    const updatedSoil = await prisma.soil.update({
      where: { id: Number(id) },
      data: { name, description, geometry },
    });
    res.json(updatedSoil);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update soil data' });
  }
};

export const deleteSoil = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.soil.delete({ where: { id: Number(id) } });
    res.json({ message: 'Soil deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete soil' });
  }
};
