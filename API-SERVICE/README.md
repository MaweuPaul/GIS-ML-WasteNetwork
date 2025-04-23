# GIS-ML-WasteNetwork API Service

This directory contains the backend API service for the GIS-ML-WasteNetwork platform. It handles data management, GIS queries, and business logic for the waste management system.

## Architecture

The API service follows a modular architecture with the following components:

- **Controllers**: Handle HTTP requests and responses
- **Routes**: Define API endpoints
- **Services**: Implement business logic
- **Middleware**: Process requests before they reach route handlers
- **Prisma ORM**: Manages database connections and queries
- **Utils**: Utility functions for common operations

## API Endpoints

The API provides endpoints for various GIS and waste management operations:

### GIS Data Endpoints

- `/api/soils`: Get and manage soil data
- `/api/protected-areas`: Get and manage protected area boundaries
- `/api/rivers`: Get and manage river data
- `/api/roads`: Get and manage road network data
- `/api/digital-elevation-models`: Get and manage elevation data
- `/api/area-of-interest`: Get and manage AOI boundaries
- `/api/land-use-raster`: Get and manage land use classification data
- `/api/settlement`: Get and manage settlement data
- `/api/geology`: Get and manage geological data

### Waste Management Endpoints

- `/api/waste-management/landfills`: Get landfill sites
- `/api/waste-management/collection-points`: Get collection points
- `/api/waste-management/summary`: Get statistical summary
- `/api/waste-management/routes`: Save optimized routes

### Operational Endpoints

- `/api/incidents`: Report and manage waste-related incidents
- `/api/specialPickup`: Request and manage special waste pickups
- `/api/zones`: Manage collection zones
- `/api/collection-schedule`: Manage waste collection schedules

### Data Management Endpoints

- `/api/database`: Manage database operations
- `/api/data`: Upload, validate, and process datasets

## Data Models

The API uses a PostgreSQL database with PostGIS extension. Key models include:

- **GIS Models**: Soil, River, Road, ProtectedArea, Settlement, Geology, DigitalElevationModel
- **Operational Models**: CollectionSchedule, IncidentReport, SpecialPickup, Zone
- **Waste Management Models**: CollectionPoint, LandfillSite, Route

## Getting Started

### Prerequisites

- Node.js (v14+)
- PostgreSQL with PostGIS extension
- Environment variables configured

### Installation

1. Install dependencies:

```bash
npm install
```

2. Configure environment variables:
   Create a `.env` file in the API-SERVICE directory with:

```
DATABASE_URL="postgresql://username:password@localhost:5432/waste_management?schema=public"
PORT=3000
CLIENT_URL="http://localhost:5173"
```

3. Generate Prisma client:

```bash
npx prisma generate
```

4. Run database migrations:

```bash
npx prisma migrate dev
```

### Running the API

```bash
npm run dev
```

The API will be available at http://localhost:3000.

## API Documentation

### Request and Response Format

All API endpoints follow a consistent format:

**Success Response**:

```json
{
  "success": true,
  "data": [{}],
  "count": 1
}
```

**Error Response**:

```json
{
  "success": false,
  "error": "Error message"
}
```

### Authentication

Authentication functionality can be implemented using JWT tokens.

## Development

### Adding New Endpoints

1. Create a controller in the `Controllers` directory
2. Create a service in the `services` directory
3. Define routes in the `Routes` directory
4. Register routes in `server.js`

### Database Schema Updates

Update the Prisma schema in `prisma/schema.prisma` and run:

```bash
npx prisma migrate dev --name descriptive_name
```

## Dependencies

The API service relies on the following key dependencies:

- **express**: Web framework
- **prisma**: ORM for database operations
- **cors**: Cross-origin resource sharing
- **dotenv**: Environment variable management
- **multer**: File upload handling
- **geojson**: GeoJSON handling

See `package.json` for the complete list of dependencies.
