const express = require('express');
const {
  getSoils,
  getSoil,
  updateSoil,
  deleteSoil,
  createSoils,
  deleteAllSoils,
} = require('../controllers/soilContoller');

const router = express.Router();

router.get('/', getSoils);
router.get('/:id', getSoil);
router.post('/', createSoils);
router.put('/:id', updateSoil);
router.delete('/deleteAll', deleteAllSoils);
router.delete('/:id', deleteSoil);

module.exports = router;
