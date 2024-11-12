const { Prisma } = require('@prisma/client');
const prisma = require('../Lib/prisma.js');

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
  async saveRoutes(routesData) {
    try {
      const savedRoutes = [];

      for (const route of routesData) {
        // Validate required fields
        if (
          !route.collection_point_id ||
          !route.landfill_id ||
          !route.distance_meters ||
          !route.geometry
        ) {
          throw new Error('Missing required fields in route data');
        }

        // Convert geometry to PostGIS format
        const geomSQL = Prisma.sql`ST_SetSRID(ST_GeomFromGeoJSON(${JSON.stringify(
          route.geometry
        )}), 21037)`;

        // Create the route record
        const savedRoute = await prisma.$queryRaw`
          INSERT INTO routes (
            collection_point_id,
            landfill_site_id,
            distance_meters,
            geom,
            created_at,
            updated_at
          )
          VALUES (
            ${parseInt(route.collection_point_id)},
            ${parseInt(route.landfill_id)},
            ${parseFloat(route.distance_meters)},
            ${geomSQL},
            NOW(),
            NOW()
          )
          RETURNING id, collection_point_id, landfill_site_id, distance_meters;
        `;

        console.log('Route saved:', savedRoute);
        savedRoutes.push(savedRoute[0]);
      }

      return {
        success: true,
        count: savedRoutes.length,
        routes: savedRoutes,
      };
    } catch (error) {
      console.error('Error saving routes:', error);
      throw error;
    }
  },
};

module.exports = wasteManagementService;
