const prisma = require('../lib/prisma');

export const getDigitalElevationModels = async (req, res) => {
  try {
    const digitalElevationModels =
      await prisma.digitalElevationModel.findMany();
    res.json(digitalElevationModels);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to fetch digital elevation models' });
  }
};

export const getDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  try {
    const digitalElevationModel = await prisma.digitalElevationModel.findUnique(
      { where: { id: Number(id) } }
    );
    res.json(digitalElevationModel);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to fetch digital elevation model' });
  }
};

export const createDigitalElevationModel = async (req, res) => {
  const { name, resolution, geometry } = req.body;
  try {
    const newDigitalElevationModel = await prisma.digitalElevationModel.create({
      data: { name, resolution, geometry },
    });
    res
      .status(200)
      .json({
        message: 'Digital elevation model data added',
        newDigitalElevationModel,
      });
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to create digital elevation model data' });
  }
};

export const updateDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  const { name, resolution, geometry } = req.body;
  try {
    const updatedDigitalElevationModel =
      await prisma.digitalElevationModel.update({
        where: { id: Number(id) },
        data: { name, resolution, geometry },
      });
    res.json(updatedDigitalElevationModel);
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to update digital elevation model data' });
  }
};

export const deleteDigitalElevationModel = async (req, res) => {
  const { id } = req.params;
  try {
    await prisma.digitalElevationModel.delete({ where: { id: Number(id) } });
    res.json({ message: 'Digital elevation model deleted' });
  } catch (error) {
    console.error(error);
    res
      .status(500)
      .json({ message: 'Failed to delete digital elevation model' });
  }
};
