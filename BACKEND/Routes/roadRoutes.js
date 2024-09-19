const express = require('express');
const {
  createRoad,
  deleteRoad,
  getRoad,
  getRoads,
  updateRoad,
  deleteAllRoads,
} = require('../controllers/roadController');

const router = express.Router();

// Middleware to parse JSON data
router.use(express.json({ limit: '10mb' }));

// Routes
router.get('/', getRoads);
router.get('/:id', getRoad);
router.post('/', createRoad);
router.put('/:id', updateRoad);
router.delete('/:id', deleteRoad);
router.delete('/deleteAll', deleteAllRoads);

// Error handling middleware
router.use((error, req, res, next) => {
  console.error('Router error:', error);
  res
    .status(500)
    .json({ message: 'An unknown error occurred', error: error.message });
});

module.exports = router;
