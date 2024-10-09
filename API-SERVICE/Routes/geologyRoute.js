const express = require('express');
const {
  getGeologies,
  getGeology,
  createGeology,
  updateGeology,
  deleteGeology,
  deleteAllGeology,
} = require('../Controllers/geologyController.js');

const router = express.Router();

router.get('/', getGeologies);
router.get('/:id', getGeology);
router.post('/', createGeology);
router.put('/:id', updateGeology);
router.delete('/deleteAll', deleteAllGeology);
router.delete('/:id', deleteGeology);

module.exports = router;
