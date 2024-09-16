const express = require('express');
const {
  getGeologies,
  getGeology,
  createGeology,
  updateGeology,
  deleteGeology,
} = require('../Controllers/geologyController.js');
const multer = require('multer');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.get('/', getGeologies);
router.get('/:id', getGeology);
router.post('/', upload.single('file'), createGeology);
router.put('/:id', updateGeology);
router.delete('/:id', deleteGeology);

module.exports = router;
