# GIS-ML-WasteNetwork API Documentation

This document provides detailed information about the GIS-ML-WasteNetwork API endpoints, request/response formats, and error handling.

## Table of Contents

1. [Authentication](#authentication)
2. [GIS Data Endpoints](#gis-data-endpoints)
3. [Waste Management Endpoints](#waste-management-endpoints)
4. [Incident Reporting Endpoints](#incident-reporting-endpoints)
5. [Special Pickup Endpoints](#special-pickup-endpoints)
6. [Zone Management Endpoints](#zone-management-endpoints)
7. [Data Management Endpoints](#data-management-endpoints)
8. [Error Codes](#error-codes)

## Authentication

_Note: Authentication implementation details would be specified here._

## GIS Data Endpoints

### Soil Data

#### Get All Soil Data

```
GET /api/soils
```

**Query Parameters:**

- `bbox` (optional): Bounding box for spatial filtering (format: minX,minY,maxX,maxY)
- `limit` (optional): Number of records to return (default: 100)
- `offset` (optional): Number of records to skip (default: 0)

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "soilType": "Clay",
      "properties": { "ph": 6.5, "organicMatter": "medium" },
      "bbox": [35.2, -1.3, 35.3, -1.2],
      "geom": { "type": "Polygon", "coordinates": [...] }
    }
  ],
  "count": 1
}
```

#### Get Soil Data By ID

```
GET /api/soils/:id
```

**Response:**

```json
{
  "success": true,
  "data": {
    "id": 1,
    "soilType": "Clay",
    "properties": { "ph": 6.5, "organicMatter": "medium" },
    "bbox": [35.2, -1.3, 35.3, -1.2],
    "geom": { "type": "Polygon", "coordinates": [...] }
  }
}
```

### Protected Areas

#### Get All Protected Areas

```
GET /api/protected-areas
```

**Query Parameters:**

- `bbox` (optional): Bounding box for spatial filtering
- `type` (optional): Filter by protection type (e.g., "NATIONAL_PARK", "RESERVE")

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Sample National Park",
      "type": "NATIONAL_PARK",
      "bbox": [35.2, -1.3, 35.3, -1.2],
      "geom": { "type": "Polygon", "coordinates": [...] }
    }
  ],
  "count": 1
}
```

### Roads

#### Get All Roads

```
GET /api/roads
```

**Query Parameters:**

- `bbox` (optional): Bounding box for spatial filtering
- `type` (optional): Filter by road type (e.g., "PRIMARY", "SECONDARY")

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Main Highway",
      "type": "PRIMARY",
      "bbox": [35.2, -1.3, 35.3, -1.2],
      "geom": { "type": "LineString", "coordinates": [...] }
    }
  ],
  "count": 1
}
```

### Digital Elevation Model

#### Get Elevation Data

```
GET /api/digital-elevation-models
```

**Query Parameters:**

- `bbox` (required): Bounding box for spatial filtering
- `resolution` (optional): Desired resolution in meters (default: 30)

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "DEM Tile 1",
      "elevation": 1200,
      "bbox": [35.2, -1.3, 35.3, -1.2],
      "geom": { "type": "Point", "coordinates": [35.25, -1.25] }
    }
  ],
  "count": 1
}
```

### Area of Interest

#### Get All Areas of Interest

```
GET /api/area-of-interest
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Project Area 1",
      "bbox": [35.2, -1.3, 35.3, -1.2],
      "geom": { "type": "Polygon", "coordinates": [...] }
    }
  ],
  "count": 1
}
```

### Settlements

#### Get All Settlements

```
GET /api/settlement
```

**Query Parameters:**

- `bbox` (optional): Bounding box for spatial filtering
- `type` (optional): Filter by settlement type (e.g., "URBAN", "RURAL")

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Downtown",
      "type": "URBAN",
      "geom": { "type": "Point", "coordinates": [35.25, -1.25] }
    }
  ],
  "count": 1
}
```

## Waste Management Endpoints

### Landfill Sites

#### Get All Landfill Sites

```
GET /api/waste-management/landfills
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Central Landfill",
      "capacity": 5000,
      "location": { "type": "Point", "coordinates": [36.82, -1.29] },
      "status": "ACTIVE",
      "wasteTypes": ["GENERAL", "RECYCLABLE"]
    }
  ],
  "count": 1
}
```

### Collection Points

#### Get All Collection Points

```
GET /api/waste-management/collection-points
```

**Query Parameters:**

- `bbox` (optional): Bounding box for spatial filtering
- `status` (optional): Filter by status (e.g., "ACTIVE", "INACTIVE")

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "point_id": 101,
      "description": "Corner of Main St",
      "geom": { "type": "Point", "coordinates": [36.82, -1.29] }
    }
  ],
  "count": 1
}
```

#### Get Summary Statistics

```
GET /api/waste-management/summary
```

**Response:**

```json
{
  "success": true,
  "data": {
    "totalLandfills": 5,
    "totalCollectionPoints": 120,
    "totalRoutes": 25,
    "activeIncidents": 12,
    "pendingPickups": 7
  }
}
```

#### Save Optimized Routes

```
POST /api/waste-management/routes
```

**Request Body:**

```json
{
  "routes": [
    {
      "name": "Route 1",
      "description": "City Center Route",
      "wasteType": "GENERAL",
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [36.82, -1.29],
          [36.83, -1.28],
          [36.84, -1.27]
        ]
      },
      "collectionPoints": [1, 2, 3]
    }
  ]
}
```

**Response:**

```json
{
  "success": true,
  "message": "Successfully saved 1 routes",
  "data": {
    "count": 1,
    "routeIds": [5]
  }
}
```

## Incident Reporting Endpoints

#### Report New Incident

```
POST /api/incidents
```

**Request Body:**

```json
{
  "type": "ILLEGAL_DUMPING",
  "description": "Large pile of construction waste",
  "priority": "HIGH",
  "location": {
    "type": "Point",
    "coordinates": [36.82, -1.29]
  },
  "contactName": "John Doe",
  "contactPhone": "1234567890",
  "contactEmail": "john@example.com"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Incident reported successfully",
  "data": {
    "id": 1,
    "type": "ILLEGAL_DUMPING",
    "status": "PENDING"
  }
}
```

#### Get All Incidents

```
GET /api/incidents
```

**Query Parameters:**

- `status` (optional): Filter by status (PENDING, IN_PROGRESS, RESOLVED)
- `type` (optional): Filter by incident type
- `priority` (optional): Filter by priority

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "type": "ILLEGAL_DUMPING",
      "description": "Large pile of construction waste",
      "priority": "HIGH",
      "status": "PENDING",
      "location": {
        "type": "Point",
        "coordinates": [36.82, -1.29]
      },
      "createdAt": "2023-05-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

## Special Pickup Endpoints

#### Request Special Pickup

```
POST /api/specialPickup
```

**Request Body:**

```json
{
  "name": "Jane Doe",
  "phone": "1234567890",
  "email": "jane@example.com",
  "address": "123 Main St",
  "location": {
    "type": "Point",
    "coordinates": [36.82, -1.29]
  },
  "wasteType": "BULKY",
  "description": "Old furniture removal",
  "preferredDate": "2023-06-01",
  "preferredTimeSlot": "MORNING"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Special pickup request submitted successfully",
  "data": {
    "id": 1,
    "status": "PENDING",
    "referenceNumber": "SP-20230515-001"
  }
}
```

## Zone Management Endpoints

#### Get All Zones

```
GET /api/zones
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Zone A",
      "description": "Central Business District",
      "geom": {
        "type": "Polygon",
        "coordinates": [...]
      },
      "collectionDays": ["MONDAY", "THURSDAY"]
    }
  ],
  "count": 1
}
```

## Data Management Endpoints

#### Upload GIS Dataset

```
POST /api/data/upload
```

**Request:**
Form data with file upload

**Response:**

```json
{
  "success": true,
  "message": "File uploaded successfully",
  "data": {
    "fileId": "12345",
    "fileName": "roads.geojson",
    "fileSize": 1024,
    "uploadDate": "2023-05-15T10:30:00Z"
  }
}
```

## Error Codes

The API uses standard HTTP status codes and returns error messages in a consistent format:

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

Common error codes:

- `INVALID_INPUT`: The request contains invalid parameters
- `RESOURCE_NOT_FOUND`: The requested resource does not exist
- `UNAUTHORIZED`: Authentication is required or failed
- `FORBIDDEN`: The user does not have permission
- `INTERNAL_ERROR`: An unexpected server error occurred
- `DATABASE_ERROR`: A database operation failed
- `VALIDATION_ERROR`: Request validation failed
- `FILE_UPLOAD_ERROR`: File upload failed
