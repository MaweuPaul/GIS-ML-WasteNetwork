const express = require('express');
const {
  createRoad,
  deleteRoad,
  getRoad,
  getRoads,
  updateRoad,
} = require('../Controllers/roadController.js');
const multer = require('multer');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.get('/', getRoads);
router.get('/:id', getRoad);
router.post('/', upload.single('file'), createRoad);
router.put('/:id', updateRoad);
router.delete('/:id', deleteRoad);

module.exports = router;
