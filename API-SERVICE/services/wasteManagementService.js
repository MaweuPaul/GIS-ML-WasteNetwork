// services/wasteManagementService.js
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const wasteManagementService = {
  getSummaryStats: async () => {
    try {
      await prisma.$connect();

      const [landfillCount, collectionPointCount, routeCount] =
        await Promise.all([
          prisma.landfillSite.count().catch((e) => {
            console.error('Error counting landfills:', e);
            return 0;
          }),
          prisma.collectionPoint.count().catch((e) => {
            console.error('Error counting collection points:', e);
            return 0;
          }),
          prisma.route.count().catch((e) => {
            console.error('Error counting routes:', e);
            return 0;
          }),
        ]);

      // Get route statistics
      const routeStats = await prisma.route
        .aggregate({
          _avg: {
            distanceMeters: true,
          },
          _max: {
            distanceMeters: true,
          },
          _min: {
            distanceMeters: true,
          },
        })
        .catch((e) => {
          console.error('Error getting route stats:', e);
          return {
            _avg: { distanceMeters: 0 },
            _max: { distanceMeters: 0 },
            _min: { distanceMeters: 0 },
          };
        });

      // Get incident counts
      const incidentCount = await prisma.incidentReport.count().catch((e) => {
        console.error('Error counting incidents:', e);
        return 0;
      });

      // Get special pickup counts
      const specialPickupCount = await prisma.specialPickup
        .count()
        .catch((e) => {
          console.error('Error counting special pickups:', e);
          return 0;
        });

      return {
        landfillCount,
        collectionPointCount,
        routeCount,
        incidentCount,
        specialPickupCount,
        routeStats: {
          averageDistance: routeStats._avg.distanceMeters || 0,
          maxDistance: routeStats._max.distanceMeters || 0,
          minDistance: routeStats._min.distanceMeters || 0,
        },
      };
    } catch (error) {
      console.error('Error in getSummaryStats:', error);
      throw new Error('Failed to fetch summary statistics');
    } finally {
      await prisma.$disconnect();
    }
  },

  getLandfills: async () => {
    try {
      return await prisma.landfillSite.findMany({
        select: {
          id: true,
          landfillId: true,
          suitabilityScore: true,
          suitabilityClass: true,
          geom: true,
          routes: {
            select: {
              id: true,
              distanceMeters: true,
            },
          },
        },
      });
    } catch (error) {
      console.error('Error fetching landfills:', error);
      throw error;
    }
  },

  getCollectionPoints: async () => {
    try {
      return await prisma.collectionPoint.findMany({
        select: {
          id: true,
          pointId: true,
          description: true,
          geom: true,
          routes: {
            select: {
              id: true,
              distanceMeters: true,
              landfillSite: {
                select: {
                  landfillId: true,
                  suitabilityScore: true,
                },
              },
            },
          },
        },
      });
    } catch (error) {
      console.error('Error fetching collection points:', error);
      throw error;
    }
  },
};

module.exports = wasteManagementService;
