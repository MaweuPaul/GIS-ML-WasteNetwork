const prisma = require('../lib/prisma');

export const getProtectedAreas = async (req, res) => {
  try {
    const protectedAreas = await prisma.protectedArea.findMany();
    res.json(protectedAreas);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch protected areas' });
  }
};

export const getProtectedArea = async (req, res) => {
  const { id } = req.params;
  try {
    const protectedArea = await prisma.protectedArea.findUnique({
      where: { id: Number(id) },
    });
    res.json(protectedArea);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch protected area' });
  }
};

export const createProtectedArea = async (req, res) => {
  const { name, description, area, geometry } = req.body;
  try {
    const newProtectedArea = await prisma.protectedArea.create({
      data: { name, description, area, geometry },
    });
    res
      .status(200)
      .json({ message: 'Protected area data added', newProtectedArea });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to create protected area data' });
  }
};

export const updateProtectedArea = async (req, res) => {
  const { id } = req.params;
  const { name, description, area, geometry } = req.body;
  try {
    const updatedProtectedArea = await prisma.protectedArea.update({
      where: { id: Number(id) },
      data: { name, description, area, geometry },
    });
    res.json(updatedProtectedArea);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update protected area data' });
  }
};

export const deleteProtectedArea = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.protectedArea.delete({ where: { id: Number(id) } });
    res.json({ message: 'Protected area deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete protected area' });
  }
};
