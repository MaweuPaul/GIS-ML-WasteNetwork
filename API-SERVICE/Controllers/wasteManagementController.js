// controllers/wasteManagementController.js
const wasteManagementService = require('../services/wasteManagementService');

const wasteManagementController = {
  async getLandfills(req, res) {
    try {
      const landfills = await wasteManagementService.getLandfills();
      res.json({
        success: true,
        data: landfills,
        count: landfills.length,
      });
    } catch (error) {
      console.error('Error fetching landfills:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch landfill sites',
      });
    }
  },

  async getCollectionPoints(req, res) {
    try {
      const collectionPoints =
        await wasteManagementService.getCollectionPoints();
      res.json({
        success: true,
        data: collectionPoints,
        count: collectionPoints.length,
      });
    } catch (error) {
      console.error('Error fetching collection points:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch collection points',
      });
    }
  },

  async getSummary(req, res) {
    try {
      const summaryStats = await wasteManagementService.getSummaryStats();
      res.json({
        success: true,
        data: summaryStats,
      });
    } catch (error) {
      console.error('Error fetching summary:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch summary statistics',
      });
    }
  },
};

module.exports = wasteManagementController;
