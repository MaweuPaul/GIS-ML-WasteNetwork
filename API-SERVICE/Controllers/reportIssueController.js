const { PrismaClient, Prisma } = require('@prisma/client');
const prisma = new PrismaClient();
const fileUtils = require('../utils/fileUtils');

const incidentController = {
  createIncident: async (req, res) => {
    try {
      const {
        type,
        description,
        priority,
        location,
        contactName,
        contactPhone,
        contactEmail,
      } = req.body;

      // Parse location if it's a string
      const parsedLocation =
        typeof location === 'string' ? JSON.parse(location) : location;

      // Create using raw SQL with proper type casting
      const [incident] = await prisma.$queryRaw`
            INSERT INTO "IncidentReport" (
              type,
              description,
              priority,
              location,
              geom,
              photos,
              "contactName",
              "contactPhone",
              "contactEmail",
              status,
              "createdAt",
              "updatedAt"
            ) VALUES (
              ${type}::"IncidentType",
              ${description},
              ${priority}::"IssuePriority",
              ${JSON.stringify(parsedLocation)}::jsonb,
              ST_SetSRID(ST_MakePoint(${parsedLocation.lng}, ${
        parsedLocation.lat
      }), 4326),
              ${req.files ? req.files.map((file) => file.path) : []}::text[],
              ${contactName},
              ${contactPhone},
              ${contactEmail},
              'PENDING'::"IncidentStatus",
              NOW(),
              NOW()
            )
            RETURNING 
              id,
              type,
              description,
              priority,
              status,
              location,
              ST_AsText(geom) as geom,
              photos,
              "contactName",
              "contactPhone",
              "contactEmail",
              "createdAt",
              "updatedAt"
          `;

      res.status(201).json(incident);
    } catch (error) {
      console.error('Error creating incident report:', error);
      res.status(400).json({
        error: error.message,
        details: error.stack,
      });
    }
  },

  getAllIncidents: async (req, res) => {
    try {
      const { type, status, priority, page = 1, limit = 10 } = req.query;
      const offset = (page - 1) * limit;

      let whereClause = '';
      const conditions = [];

      if (type) conditions.push(`type = '${type}'::"IncidentType"`);
      if (status) conditions.push(`status = '${status}'::"IncidentStatus"`);
      if (priority)
        conditions.push(`priority = '${priority}'::"IssuePriority"`);

      if (conditions.length > 0) {
        whereClause = `WHERE ${conditions.join(' AND ')}`;
      }

      const incidents = await prisma.$queryRaw`
        SELECT 
          id,
          type,
          description,
          priority,
          status,
          location,
          ST_AsText(geom) as geom,
          photos,
          "contactName",
          "contactPhone",
          "contactEmail",
          "createdAt",
          "updatedAt"
        FROM "IncidentReport"
        ${Prisma.raw(whereClause)}
        ORDER BY "createdAt" DESC
        LIMIT ${parseInt(limit)}
        OFFSET ${offset};
      `;

      const totalCount = await prisma.$queryRaw`
        SELECT COUNT(*) as count
        FROM "IncidentReport"
        ${Prisma.raw(whereClause)};
      `;

      // Modified response structure to match frontend expectations
      res.json({
        incidents: incidents.map((incident) => ({
          ...incident,
          location: {
            lat: incident.location.lat,
            lng: incident.location.lng,
          },
        })),
        total: parseInt(totalCount[0].count),
        totalPages: Math.ceil(parseInt(totalCount[0].count) / limit),
      });
    } catch (error) {
      console.error('Error fetching incidents:', error);
      res.status(500).json({ error: error.message });
    }
  },

  // Get incident by ID
  getIncidentById: async (req, res) => {
    try {
      const { id } = req.params;
      const incident = await prisma.$queryRaw`
        SELECT 
          id,
          type,
          description,
          priority,
          status,
          location,
          ST_AsText(geom) as geom,
          photos,
          "contactName",
          "contactPhone",
          "contactEmail",
          "createdAt",
          "updatedAt"
        FROM "IncidentReport"
        WHERE id = ${parseInt(id)};
      `;

      if (!incident[0]) {
        return res.status(404).json({ error: 'Incident report not found' });
      }

      res.json(incident[0]);
    } catch (error) {
      console.error('Error fetching incident:', error);
      res.status(500).json({ error: error.message });
    }
  },

  // Update incident status
  updateIncidentStatus: async (req, res) => {
    try {
      const { id } = req.params;
      const { status } = req.body;

      // Validate status
      const validStatuses = ['PENDING', 'IN_PROGRESS', 'RESOLVED'];
      if (!validStatuses.includes(status)) {
        return res.status(400).json({ error: 'Invalid status value' });
      }

      const incident = await prisma.$queryRaw`
        UPDATE "IncidentReport"
        SET 
          status = ${status}::"IncidentStatus",
          "updatedAt" = NOW()
        WHERE id = ${parseInt(id)}
        RETURNING 
          id,
          type,
          description,
          priority,
          status,
          location,
          ST_AsText(geom) as geom,
          photos,
          "contactName",
          "contactPhone",
          "contactEmail",
          "createdAt",
          "updatedAt"
      `;

      if (!incident[0]) {
        return res.status(404).json({ error: 'Incident not found' });
      }

      res.json(incident[0]);
    } catch (error) {
      console.error('Error updating incident status:', error);
      res.status(400).json({ error: error.message });
    }
  },

  // Get nearby incidents
  getNearbyIncidents: async (req, res) => {
    try {
      const { lat, lng, distance } = req.params; // distance in meters
      const incidents = await prisma.$queryRaw`
        SELECT 
          id,
          type,
          description,
          priority,
          status,
          location,
          ST_AsText(geom) as geom,
          photos,
          "contactName",
          "contactPhone",
          "contactEmail",
          "createdAt",
          "updatedAt",
          ST_Distance(
            geom::geography,
            ST_SetSRID(ST_MakePoint(${parseFloat(lng)}, ${parseFloat(
        lat
      )})::geography, 4326)
          ) as distance
        FROM "IncidentReport"
        WHERE ST_DWithin(
          geom::geography,
          ST_SetSRID(ST_MakePoint(${parseFloat(lng)}, ${parseFloat(
        lat
      )})::geography, 4326),
          ${parseFloat(distance)}
        )
        ORDER BY distance;
      `;
      res.json(incidents);
    } catch (error) {
      console.error('Error fetching nearby incidents:', error);
      res.status(500).json({ error: error.message });
    }
  },

  deleteIncident: async (req, res) => {
    try {
      const { id } = req.params;

      // First, get the incident to retrieve file paths
      const incident = await prisma.incidentReport.findUnique({
        where: { id: parseInt(id) },
        select: { photos: true },
      });

      if (!incident) {
        return res.status(404).json({ error: 'Incident report not found' });
      }

      // Delete the incident from database
      await prisma.incidentReport.delete({
        where: { id: parseInt(id) },
      });

      // Clean up associated files
      if (incident.photos && incident.photos.length > 0) {
        await fileUtils.deleteFiles(incident.photos);
      }

      // Modified response to match frontend expectations
      res.json({
        success: true,
        message: 'Incident deleted successfully',
      });
    } catch (error) {
      console.error('Error deleting incident:', error);
      res.status(400).json({ error: error.message });
    }
  },

  // Update incident with file cleanup for replaced photos
  updateIncident: async (req, res) => {
    try {
      const { id } = req.params;
      const {
        type,
        description,
        priority,
        location,
        contactName,
        contactPhone,
        contactEmail,
      } = req.body;

      // Get the existing incident to check for files to delete
      const existingIncident = await prisma.incidentReport.findUnique({
        where: { id: parseInt(id) },
        select: { photos: true },
      });

      if (!existingIncident) {
        return res.status(404).json({ error: 'Incident report not found' });
      }

      // Parse location if it's a string
      const parsedLocation =
        typeof location === 'string' ? JSON.parse(location) : location;

      // Prepare update data
      const updateData = {
        type,
        description,
        priority,
        location: parsedLocation,
        contactName,
        contactPhone,
        contactEmail,
      };

      // If new files are uploaded, update photos
      if (req.files && req.files.length > 0) {
        updateData.photos = req.files.map((file) => file.path);
        // Clean up old files
        await fileUtils.deleteFiles(existingIncident.photos);
      }

      // Update the incident
      const updatedIncident = await prisma.incidentReport.update({
        where: { id: parseInt(id) },
        data: updateData,
      });

      res.json(updatedIncident);
    } catch (error) {
      console.error('Error updating incident:', error);
      res.status(400).json({ error: error.message });
    }
  },

  // Clean up files for multiple incidents
  cleanupIncidentFiles: async (req, res) => {
    try {
      const { ids } = req.body;

      // Get all incidents with their photos
      const incidents = await prisma.incidentReport.findMany({
        where: {
          id: {
            in: ids.map((id) => parseInt(id)),
          },
        },
        select: {
          id: true,
          photos: true,
        },
      });

      // Collect all file paths
      const allPhotos = incidents.reduce(
        (acc, incident) => [...acc, ...incident.photos],
        []
      );

      // Delete all files
      await fileUtils.deleteFiles(allPhotos);

      // Delete incidents from database
      await prisma.incidentReport.deleteMany({
        where: {
          id: {
            in: ids.map((id) => parseInt(id)),
          },
        },
      });

      res.json({
        message: 'Incidents and associated files deleted successfully',
        deletedFiles: allPhotos,
      });
    } catch (error) {
      console.error('Error cleaning up incidents:', error);
      res.status(400).json({ error: error.message });
    }
  },
};

module.exports = incidentController;
