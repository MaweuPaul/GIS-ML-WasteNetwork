const prisma = require('../lib/prisma');

export const getGeologies = async (req, res) => {
  try {
    const geologies = await prisma.geology.findMany();
    res.json(geologies);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch geologies' });
  }
};

export const getGeology = async (req, res) => {
  const { id } = req.params;
  try {
    const geology = await prisma.geology.findUnique({
      where: { id: Number(id) },
    });
    res.json(geology);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch geology' });
  }
};

export const createGeology = async (req, res) => {
  const { type, description, geometry } = req.body;
  try {
    const newGeology = await prisma.geology.create({
      data: { type, description, geometry },
    });
    res.status(200).json({ message: 'Geology data added', newGeology });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to create geology data' });
  }
};

export const updateGeology = async (req, res) => {
  const { id } = req.params;
  const { type, description, geometry } = req.body;
  try {
    const updatedGeology = await prisma.geology.update({
      where: { id: Number(id) },
      data: { type, description, geometry },
    });
    res.json(updatedGeology);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update geology data' });
  }
};

export const deleteGeology = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.geology.delete({ where: { id: Number(id) } });
    res.json({ message: 'Geology deleted' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to delete geology' });
  }
};
