const express = require('express');
const {
  createDigitalElevationModel,
  deleteDigitalElevationModel,
  getDigitalElevationModel,
  getDigitalElevationModels,
  updateDigitalElevationModel,
  deleteAllDigitalElevationModel,
} = require('../Controllers/demController');

const router = express.Router();

router.get('/', getDigitalElevationModels);
router.get('/:id', getDigitalElevationModel);
router.post('/', createDigitalElevationModel);
router.put('/:id', updateDigitalElevationModel);
router.delete('/deleteAll', deleteAllDigitalElevationModel);
router.delete('/:id', deleteDigitalElevationModel);

module.exports = router;
