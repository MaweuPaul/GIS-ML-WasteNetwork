// routes/admin/zones.js
const express = require('express');
const router = express.Router();
const zoneController = require('../Controllers/zonesController');

// Get all zones
router.get('/', zoneController.getAllZones);

// Get single zone
router.get('/:id', zoneController.getZone);

// Create new zone
router.post('/', zoneController.createZone);

// Update zone
router.put('/:id', zoneController.updateZone);

// Delete zone
router.delete('/:id', zoneController.deleteZone);

// Search zones by area
router.get('/search/:query', zoneController.searchZones);

module.exports = router;
