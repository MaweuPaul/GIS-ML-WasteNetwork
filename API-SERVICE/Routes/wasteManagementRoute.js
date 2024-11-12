const express = require('express');
const router = express.Router();
const wasteManagementController = require('../Controllers/wasteManagementController');

router.get('/landfills', wasteManagementController.getLandfills);
router.get('/collection-points', wasteManagementController.getCollectionPoints);
router.get('/summary', wasteManagementController.getSummary);
router.post('/save-routes', wasteManagementController.saveRoutes);

module.exports = router;
