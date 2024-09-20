const express = require('express');
const {
  createDigitalElevationModel,
  deleteDigitalElevationModel,
  getDigitalElevationModel,
  getDigitalElevationModels,
  updateDigitalElevationModel,
} = require('../Controllers/demController');

const router = express.Router();

router.get('/', getDigitalElevationModels);
router.get('/:id', getDigitalElevationModel);
router.post('/', createDigitalElevationModel);
router.put('/:id', updateDigitalElevationModel);
router.delete('/:id', deleteDigitalElevationModel);

module.exports = router;
