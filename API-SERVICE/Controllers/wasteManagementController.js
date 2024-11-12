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

  async saveRoutes(req, res) {
    try {
      const { routes } = req.body;

      if (!Array.isArray(routes)) {
        return res.status(400).json({
          success: false,
          error: 'Routes must be an array',
        });
      }

      const result = await wasteManagementService.saveRoutes(routes);

      res.json({
        success: true,
        message: `Successfully saved ${result.count} routes`,
        data: result,
      });
    } catch (error) {
      console.error('Error in saveRoutes controller:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to save routes',
      });
    }
  },
};

module.exports = wasteManagementController;
