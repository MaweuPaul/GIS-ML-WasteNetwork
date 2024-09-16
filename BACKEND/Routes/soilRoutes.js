const express = require('express');
const multer = require('multer');
const {
  getSoils,
  getSoil,
  createSoil,
  updateSoil,
  deleteSoil,
} = require('../controllers/soilContoller.js');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.get('/', getSoils);
router.get('/:id', getSoil);
router.post('/', upload.single('file'), createSoil);
router.put('/:id', updateSoil);
router.delete('/:id', deleteSoil);

module.exports = router;
