const express = require('express');
const riverController = require('../controllers/riverControllers');

const router = express.Router();

// Middleware to parse JSON data
router.use(express.json({ limit: '10mb' }));

// Routes
router.get('/', riverController.getRivers);
router.get('/:id', riverController.getRiver);
router.post('/', riverController.createRiver);
router.put('/:id', riverController.updateRiver);
router.delete('/deleteAll', riverController.deleteAllRivers);
router.delete('/:id', riverController.deleteRiver);

// Error handling middleware
router.use((error, req, res, next) => {
  console.error('Router error:', error);
  res
    .status(500)
    .json({ message: 'An unknown error occurred', error: error.message });
});

module.exports = router;
