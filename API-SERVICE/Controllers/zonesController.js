// controllers/zoneController.js
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const zoneController = {
  // Get all zones
  getAllZones: async (req, res) => {
    try {
      const zones = await prisma.zone.findMany({
        orderBy: {
          createdAt: 'desc',
        },
      });
      res.json(zones);
    } catch (error) {
      console.error('Error fetching zones:', error);
      res.status(500).json({ error: 'Failed to fetch zones' });
    }
  },

  // Get single zone
  getZone: async (req, res) => {
    try {
      const zone = await prisma.zone.findUnique({
        where: {
          id: parseInt(req.params.id),
        },
      });

      if (!zone) {
        return res.status(404).json({ error: 'Zone not found' });
      }

      res.json(zone);
    } catch (error) {
      console.error('Error fetching zone:', error);
      res.status(500).json({ error: 'Failed to fetch zone' });
    }
  },

  // Create new zone
  createZone: async (req, res) => {
    try {
      const { name, coordinates, areas, collectionDays, time } = req.body;

      if (!name || !coordinates || !areas || !collectionDays || !time) {
        return res.status(400).json({ error: 'All fields are required' });
      }

      const geoJsonCoordinates = {
        type: 'Polygon',
        coordinates: [coordinates],
      };

      const zone = await prisma.zone.create({
        data: {
          name,
          coordinates: geoJsonCoordinates,
          areas,
          collectionDays,
          time,
        },
      });

      res.status(201).json(zone);
    } catch (error) {
      if (error.code === 'P2002') {
        return res.status(400).json({ error: 'Zone name already exists' });
      }
      console.error('Error creating zone:', error);
      res.status(500).json({ error: 'Failed to create zone' });
    }
  },

  // Update zone
  updateZone: async (req, res) => {
    try {
      const { name, coordinates, areas, collectionDays, time } = req.body;

      if (!name || !coordinates || !areas || !collectionDays || !time) {
        return res.status(400).json({ error: 'All fields are required' });
      }

      const geoJsonCoordinates = {
        type: 'Polygon',
        coordinates: [coordinates],
      };

      const zone = await prisma.zone.update({
        where: {
          id: parseInt(req.params.id),
        },
        data: {
          name,
          coordinates: geoJsonCoordinates,
          areas,
          collectionDays,
          time,
        },
      });

      res.json(zone);
    } catch (error) {
      if (error.code === 'P2002') {
        return res.status(400).json({ error: 'Zone name already exists' });
      }
      if (error.code === 'P2025') {
        return res.status(404).json({ error: 'Zone not found' });
      }
      console.error('Error updating zone:', error);
      res.status(500).json({ error: 'Failed to update zone' });
    }
  },

  // Delete zone
  deleteZone: async (req, res) => {
    try {
      await prisma.zone.delete({
        where: {
          id: parseInt(req.params.id),
        },
      });

      res.json({ message: 'Zone deleted successfully' });
    } catch (error) {
      if (error.code === 'P2025') {
        return res.status(404).json({ error: 'Zone not found' });
      }
      console.error('Error deleting zone:', error);
      res.status(500).json({ error: 'Failed to delete zone' });
    }
  },

  // Search zones
  searchZones: async (req, res) => {
    try {
      const { query } = req.params;
      const zones = await prisma.zone.findMany({
        where: {
          areas: {
            hasSome: [query],
          },
        },
      });
      res.json(zones);
    } catch (error) {
      console.error('Error searching zones:', error);
      res.status(500).json({ error: 'Failed to search zones' });
    }
  },
};

module.exports = zoneController;
