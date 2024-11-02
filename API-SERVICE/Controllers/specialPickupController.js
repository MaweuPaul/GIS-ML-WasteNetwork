const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const specialPickupController = {
  // Create new special pickup request
  async create(req, res) {
    try {
      const {
        type,
        quantity,
        description,
        preferredDate,
        preferredTime,
        location,
        contactName,
        contactPhone,
        contactEmail,
      } = req.body;

      // Handle file uploads if any
      const photoUrls = req.files ? req.files.map((file) => file.path) : [];

      // Create point geometry from coordinates
      const point = `POINT(${location.lng} ${location.lat})`;

      const pickup = await prisma.specialPickup.create({
        data: {
          type,
          quantity,
          description,
          preferredDate: new Date(preferredDate),
          preferredTime,
          location,
          geom: point,
          photos: photoUrls,
          contactName,
          contactPhone,
          contactEmail,
        },
      });

      res.status(201).json(pickup);
    } catch (error) {
      console.error('Error creating special pickup:', error);
      res
        .status(400)
        .json({ error: 'Failed to create special pickup request' });
    }
  },

  // Get pickup request by ID
  async track(req, res) {
    try {
      const { id } = req.params;

      const pickup = await prisma.specialPickup.findUnique({
        where: { id: Number(id) },
      });

      if (!pickup) {
        return res.status(404).json({ error: 'Pickup request not found' });
      }

      res.status(200).json(pickup);
    } catch (error) {
      console.error('Error tracking pickup:', error);
      res.status(400).json({ error: 'Failed to track pickup request' });
    }
  },

  // Get nearby pickup requests
  async getNearby(req, res) {
    try {
      const { lat, lng, radius = 5000 } = req.query; // radius in meters

      const pickups = await prisma.$queryRaw`
        SELECT *
        FROM "SpecialPickup"
        WHERE ST_DWithin(
          geom,
          ST_SetSRID(ST_MakePoint(${Number(lng)}, ${Number(
        lat
      )})::geometry, 4326),
          ${Number(radius)}
        );
      `;

      res.status(200).json(pickups);
    } catch (error) {
      console.error('Error fetching nearby pickups:', error);
      res.status(400).json({ error: 'Failed to fetch nearby pickups' });
    }
  },
};

module.exports = specialPickupController;
