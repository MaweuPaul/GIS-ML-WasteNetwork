const express = require('express');
const {
  getAreasOfInterest,
  getAreaOfInterest,
  createAreaOfInterest,
  updateAreaOfInterest,
  deleteAreaOfInterest,
  deleteAllAreasOfInterest,
} = require('../controllers/aoiController');

const router = express.Router();

router.get('/', getAreasOfInterest);
router.get('/:id', getAreaOfInterest);
router.post('/', createAreaOfInterest);
router.put('/:id', updateAreaOfInterest);
router.delete('/deleteAll', deleteAllAreasOfInterest);
router.delete('/:id', deleteAreaOfInterest);

module.exports = router;
