# GIS-ML-WasteNetwork API Developer Guide

This guide provides information for developers working on the GIS-ML-WasteNetwork API service. It covers the architecture, design patterns, best practices, and instructions for extending the API.

## Architecture Overview

The API service follows a layered architecture pattern:

```
Client Request → Routes → Controllers → Services → Data Access → Database
```

- **Routes** (`/Routes`): Define API endpoints and route requests to controllers
- **Controllers** (`/Controllers`): Handle HTTP requests and responses
- **Services** (`/services`): Implement business logic and data processing
- **Data Access** (via Prisma): Handles database operations
- **Middleware** (`/middleware`): Process requests before they reach controllers
- **Utils** (`/utils`): Utility functions for common operations

## Design Patterns

### MVC Pattern

The API uses a modified Model-View-Controller (MVC) pattern:

- **Model**: Prisma schema defines data models
- **Controller**: Handles request/response
- **Service**: Contains business logic (extension of traditional MVC)

### Repository Pattern

Database operations are abstracted through Prisma ORM, acting as a repository layer.

### Middleware Pattern

Common operations like validation, authentication, and logging are implemented as middleware.

## Coding Standards

### Naming Conventions

- **Files**: camelCase for files (e.g., `soilController.js`)
- **Directories**: PascalCase for directories containing components (e.g., `Controllers/`)
- **Functions**: camelCase for functions (e.g., `getAllSoils()`)
- **Variables**: camelCase for variables (e.g., `const soilData = ...`)
- **Constants**: UPPER_SNAKE_CASE for constants (e.g., `const MAX_LIMIT = 100`)

### Error Handling

All controller methods should follow this pattern:

```javascript
try {
  // Async operations
} catch (error) {
  console.error('Error description:', error);
  res.status(500).json({
    success: false,
    error: 'User-friendly error message',
  });
}
```

### Response Format

All API responses should follow a consistent format:

```javascript
// Success response
res.json({
  success: true,
  data: result,
  count: result.length, // If applicable
});

// Error response
res.status(statusCode).json({
  success: false,
  error: 'Error message',
});
```

## Adding New Features

### Creating a New Endpoint

1. **Define the Model** (if needed):
   Add the model to `prisma/schema.prisma`

   ```prisma
   model NewResource {
     id        Int      @id @default(autoincrement())
     name      String
     createdAt DateTime @default(now())
     updatedAt DateTime @updatedAt
   }
   ```

2. **Create a Service**:
   Create a new file in the `services` directory

   ```javascript
   // services/newResourceService.js
   const { PrismaClient } = require('@prisma/client');
   const prisma = new PrismaClient();

   const newResourceService = {
     async getAll() {
       return await prisma.newResource.findMany();
     },

     async getById(id) {
       return await prisma.newResource.findUnique({
         where: { id: parseInt(id) },
       });
     },

     async create(data) {
       return await prisma.newResource.create({
         data,
       });
     },
   };

   module.exports = newResourceService;
   ```

3. **Create a Controller**:
   Create a new file in the `Controllers` directory

   ```javascript
   // Controllers/newResourceController.js
   const newResourceService = require('../services/newResourceService');

   const newResourceController = {
     async getAll(req, res) {
       try {
         const resources = await newResourceService.getAll();
         res.json({
           success: true,
           data: resources,
           count: resources.length,
         });
       } catch (error) {
         console.error('Error fetching resources:', error);
         res.status(500).json({
           success: false,
           error: 'Failed to fetch resources',
         });
       }
     },

     async getById(req, res) {
       try {
         const { id } = req.params;
         const resource = await newResourceService.getById(id);

         if (!resource) {
           return res.status(404).json({
             success: false,
             error: 'Resource not found',
           });
         }

         res.json({
           success: true,
           data: resource,
         });
       } catch (error) {
         console.error('Error fetching resource:', error);
         res.status(500).json({
           success: false,
           error: 'Failed to fetch resource',
         });
       }
     },

     async create(req, res) {
       try {
         const data = req.body;
         const resource = await newResourceService.create(data);

         res.status(201).json({
           success: true,
           message: 'Resource created successfully',
           data: resource,
         });
       } catch (error) {
         console.error('Error creating resource:', error);
         res.status(500).json({
           success: false,
           error: 'Failed to create resource',
         });
       }
     },
   };

   module.exports = newResourceController;
   ```

4. **Create Routes**:
   Create a new file in the `Routes` directory

   ```javascript
   // Routes/newResourceRoutes.js
   const express = require('express');
   const router = express.Router();
   const newResourceController = require('../Controllers/newResourceController');

   router.get('/', newResourceController.getAll);
   router.get('/:id', newResourceController.getById);
   router.post('/', newResourceController.create);

   module.exports = router;
   ```

5. **Register Routes in Server.js**:
   Add the route to `server.js`

   ```javascript
   const newResourceRoutes = require('./Routes/newResourceRoutes');

   // Add this line with other app.use statements
   app.use('/api/new-resources', newResourceRoutes);
   ```

### Working with GIS Data

When working with GIS data, follow these best practices:

1. **Validate Geometries**:
   Always validate GeoJSON geometries before storing them

2. **Use PostGIS Functions**:
   Leverage PostGIS functions for spatial operations:

   ```javascript
   const results = await prisma.$queryRaw`
     SELECT id, ST_AsGeoJSON(geom)::json as geometry 
     FROM "Road" 
     WHERE ST_Intersects(geom, ST_MakeEnvelope(${minX}, ${minY}, ${maxX}, ${maxY}, 4326))
   `;
   ```

3. **Implement Spatial Indexing**:
   Add spatial indexes to geometry columns in Prisma schema:

   ```prisma
   model Soil {
     id      Int                      @id @default(autoincrement())
     geom    Unsupported("geometry")

     @@index([geom], type: Gist)
   }
   ```

## Performance Optimization

### Database Queries

1. **Use Pagination**:
   Always implement pagination for endpoints that return lists:

   ```javascript
   const limit = parseInt(req.query.limit) || 100;
   const offset = parseInt(req.query.offset) || 0;

   const results = await prisma.soil.findMany({
     take: limit,
     skip: offset,
   });
   ```

2. **Select Only Needed Fields**:
   Specify only the required fields in queries:

   ```javascript
   const results = await prisma.soil.findMany({
     select: {
       id: true,
       soilType: true,
       geom: true,
     },
   });
   ```

3. **Use Batch Operations**:
   For multiple operations, use Prisma transactions:

   ```javascript
   const result = await prisma.$transaction(async (tx) => {
     // Multiple database operations
   });
   ```

### Caching

Implement caching for frequently accessed, relatively static data:

```javascript
const NodeCache = require('node-cache');
const cache = new NodeCache({ stdTTL: 600 }); // 10 minutes

async function getWithCache(key, fetcher) {
  const cachedData = cache.get(key);
  if (cachedData) return cachedData;

  const data = await fetcher();
  cache.set(key, data);
  return data;
}
```

## Deployment Considerations

### Environment Variables

Create a `.env` file with these variables:

```
DATABASE_URL=postgresql://username:password@localhost:5432/database
PORT=3000
NODE_ENV=development
```

### Database Migrations

Run migrations when schema changes:

```bash
npx prisma migrate dev --name descriptive_name
```

### Production Deployment

1. Set NODE_ENV to production
2. Use a process manager like PM2
3. Set up proper logging
4. Configure security headers

## Testing

### Unit Testing

Create unit tests for services:

```javascript
// tests/services/soilService.test.js
const soilService = require('../../services/soilService');
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

jest.mock('@prisma/client', () => {
  return {
    PrismaClient: jest.fn().mockImplementation(() => {
      return {
        soil: {
          findMany: jest.fn(),
        },
      };
    }),
  };
});

describe('Soil Service', () => {
  test('getAll returns all soil data', async () => {
    prisma.soil.findMany.mockResolvedValue([{ id: 1, soilType: 'Clay' }]);

    const result = await soilService.getAll();

    expect(result).toEqual([{ id: 1, soilType: 'Clay' }]);
    expect(prisma.soil.findMany).toHaveBeenCalled();
  });
});
```

### Integration Testing

Create API tests:

```javascript
// tests/api/soil.test.js
const request = require('supertest');
const app = require('../../server');

describe('Soil API', () => {
  test('GET /api/soils returns soil data', async () => {
    const response = await request(app).get('/api/soils');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(Array.isArray(response.body.data)).toBe(true);
  });
});
```

## Troubleshooting

### Common Errors

1. **Prisma Connection Issues**:

   - Check DATABASE_URL in .env
   - Verify PostgreSQL is running
   - Check firewall settings

2. **PostGIS Extensions**:

   - Ensure PostGIS is installed: `CREATE EXTENSION postgis;`

3. **CORS Issues**:
   - Check allowed origins in CORS configuration

### Debugging

Enable detailed logging:

```javascript
const prisma = new PrismaClient({
  log: ['query', 'info', 'warn', 'error'],
});
```

## Conclusion

This guide covers the core aspects of developing for the GIS-ML-WasteNetwork API service. By following these patterns and best practices, you'll ensure consistency and maintainability as the project evolves.
