const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const { PrismaClient, Prisma } = require('@prisma/client');

const prisma = new PrismaClient();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  },
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only images are allowed'));
    }
  },
});
router.post('/', upload.array('photos', 5), async (req, res) => {
  try {
    const {
      wasteType,
      quantity,
      description,
      preferredDate,
      preferredTime,
      location,
      contactName,
      contactPhone,
      contactEmail,
    } = req.body;

    // Parse location if it's a string
    const parsedLocation =
      typeof location === 'string' ? JSON.parse(location) : location;

    const now = new Date();

    // Create the special pickup request
    const specialPickup = await prisma.$queryRaw`
        INSERT INTO "SpecialPickup" (
          "type",
          "quantity",
          "description",
          "preferredDate",
          "preferredTime",
          "location",
          "contactName",
          "contactPhone",
          "contactEmail",
          "photos",
          "status",
          "geom",
          "createdAt",
          "updatedAt"
        ) VALUES (
          ${wasteType}::"WasteType",
          ${quantity},
          ${description},
          ${new Date(preferredDate)}::timestamp,
          ${preferredTime},
          ${JSON.stringify(parsedLocation)}::jsonb,
          ${contactName},
          ${contactPhone},
          ${contactEmail},
          ${req.files ? req.files.map((file) => file.path) : []}::text[],
          'PENDING'::"PickupStatus",
          ST_SetSRID(ST_MakePoint(${parsedLocation.lng}, ${
      parsedLocation.lat
    }), 4326),
          ${now}::timestamp,
          ${now}::timestamp
        )
        RETURNING 
          id,
          type,
          quantity,
          description,
          "preferredDate",
          "preferredTime",
          location,
          "contactName",
          "contactPhone",
          "contactEmail",
          photos,
          status,
          ST_AsText(geom) as geom,
          "createdAt",
          "updatedAt";
      `;

    res.status(201).json(specialPickup[0]);
  } catch (error) {
    console.error('Error creating special pickup:', error);
    res.status(400).json({
      error: error.message,
      details: error.stack,
    });
  }
});
// Get all special pickups with location data
router.get('/', async (req, res) => {
  try {
    const specialPickups = await prisma.$queryRaw`
        SELECT 
          id,
          type,
          quantity,
          description,
          "preferredDate",
          "preferredTime",
          location,
          "contactName",
          "contactPhone",
          "contactEmail",
          photos,
          status,
          ST_AsGeoJSON(geom) as geom,
          "createdAt",
          "updatedAt"
        FROM "SpecialPickup"
        ORDER BY "createdAt" DESC;
      `;
    res.json(specialPickups);
  } catch (error) {
    console.error('Error fetching special pickups:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get special pickup by id with location data
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const specialPickup = await prisma.$queryRaw`
        SELECT 
          id,
          type,
          quantity,
          description,
          "preferredDate",
          "preferredTime",
          location,
          "contactName",
          "contactPhone",
          "contactEmail",
          photos,
          status,
          ST_AsGeoJSON(geom) as geom,
          "createdAt",
          "updatedAt"
        FROM "SpecialPickup"
        WHERE id = ${parseInt(id)};
      `;

    if (!specialPickup[0]) {
      return res.status(404).json({ error: 'Special pickup not found' });
    }

    res.json(specialPickup[0]);
  } catch (error) {
    console.error('Error fetching special pickup:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get nearby pickups
router.get('/nearby/:lat/:lng/:distance', async (req, res) => {
  try {
    const { lat, lng, distance } = req.params; // distance in meters
    const nearbyPickups = await prisma.$queryRaw`
        SELECT 
          id,
          type,
          quantity,
          description,
          "preferredDate",
          "preferredTime",
          location,
          "contactName",
          "contactPhone",
          "contactEmail",
          photos,
          status,
          ST_AsGeoJSON(geom) as geom,
          ST_Distance(
            geom::geography,
            ST_SetSRID(ST_MakePoint(${parseFloat(lng)}, ${parseFloat(
      lat
    )})::geography, 4326)
          ) as distance,
          "createdAt",
          "updatedAt"
        FROM "SpecialPickup"
        WHERE ST_DWithin(
          geom::geography,
          ST_SetSRID(ST_MakePoint(${parseFloat(lng)}, ${parseFloat(
      lat
    )})::geography, 4326),
          ${parseFloat(distance)}
        )
        ORDER BY distance;
      `;
    res.json(nearbyPickups);
  } catch (error) {
    console.error('Error fetching nearby pickups:', error);
    res.status(500).json({ error: error.message });
  }
});
// Error handling middleware
router.use((error, req, res, next) => {
  console.error('Route error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message,
    stack: process.env.NODE_ENV === 'development' ? error.stack : undefined,
  });
});

module.exports = router;
