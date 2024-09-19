const express = require('express');
const {
  createDem,
  createDigitalElevationModel,
  deleteDigitalElevationModel,
  getDigitalElevationModel,
  getDigitalElevationModels,
  updateDigitalElevationModel,
} = require('../Controllers/demController.js');
const multer = require('multer');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.get('/', getDigitalElevationModels);
router.get('/:id', getDigitalElevationModel);
router.post('/', upload.single('file'), createDigitalElevationModel);
router.put('/:id', updateDigitalElevationModel);
router.delete('/:id', deleteDigitalElevationModel);

module.exports = router;
