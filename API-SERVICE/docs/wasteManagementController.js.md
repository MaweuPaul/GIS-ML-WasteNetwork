# Waste Management Controller Documentation

This file documents the waste management controller methods, their parameters, return values, and error handling.

## Methods

### getLandfills

Fetches all landfill sites from the database.

```javascript
/**
 * Retrieves all landfill sites
 * @async
 * @function getLandfills
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @returns {Promise<Response>} - JSON response with landfill sites
 * @throws {Error} If database access fails
 *
 * @example
 * // Response format
 * {
 *   "success": true,
 *   "data": [
 *     {
 *       "id": 1,
 *       "name": "Central Landfill",
 *       "capacity": 5000,
 *       "location": { "type": "Point", "coordinates": [36.82, -1.29] },
 *       "status": "ACTIVE",
 *       "wasteTypes": ["GENERAL", "RECYCLABLE"]
 *     }
 *   ],
 *   "count": 1
 * }
 */
async getLandfills(req, res) { ... }
```

### getCollectionPoints

Fetches all waste collection points from the database.

```javascript
/**
 * Retrieves all waste collection points
 * @async
 * @function getCollectionPoints
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @returns {Promise<Response>} - JSON response with collection points
 * @throws {Error} If database access fails
 *
 * @example
 * // Response format
 * {
 *   "success": true,
 *   "data": [
 *     {
 *       "id": 1,
 *       "point_id": 101,
 *       "description": "Corner of Main St",
 *       "location": { "type": "Point", "coordinates": [36.82, -1.29] }
 *     }
 *   ],
 *   "count": 1
 * }
 */
async getCollectionPoints(req, res) { ... }
```

### getSummary

Fetches summary statistics for the waste management system.

```javascript
/**
 * Retrieves summary statistics for waste management
 * @async
 * @function getSummary
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @returns {Promise<Response>} - JSON response with summary statistics
 * @throws {Error} If database access fails
 *
 * @example
 * // Response format
 * {
 *   "success": true,
 *   "data": {
 *     "totalLandfills": 5,
 *     "totalCollectionPoints": 120,
 *     "totalRoutes": 25,
 *     "activeIncidents": 12,
 *     "pendingPickups": 7
 *   }
 * }
 */
async getSummary(req, res) { ... }
```

### saveRoutes

Saves optimized waste collection routes to the database.

```javascript
/**
 * Saves optimized waste collection routes
 * @async
 * @function saveRoutes
 * @param {Request} req - Express request object with routes data
 * @param {Response} res - Express response object
 * @returns {Promise<Response>} - JSON response with save confirmation
 * @throws {Error} If validation fails or database access fails
 *
 * @example
 * // Request body format
 * {
 *   "routes": [
 *     {
 *       "name": "Route 1",
 *       "description": "City Center Route",
 *       "wasteType": "GENERAL",
 *       "geometry": {
 *         "type": "LineString",
 *         "coordinates": [
 *           [36.82, -1.29],
 *           [36.83, -1.28],
 *           [36.84, -1.27]
 *         ]
 *       },
 *       "collectionPoints": [1, 2, 3]
 *     }
 *   ]
 * }
 *
 * // Response format
 * {
 *   "success": true,
 *   "message": "Successfully saved 1 routes",
 *   "data": {
 *     "count": 1,
 *     "routeIds": [5]
 *   }
 * }
 */
async saveRoutes(req, res) { ... }
```

## Error Handling

All controller methods follow a consistent error handling pattern:

1. Try-catch blocks around async operations
2. Console error logging for debugging
3. HTTP 500 response with error message
4. Structured error response format

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

## Related Files

- **Service**: `../services/wasteManagementService.js`
- **Routes**: `../Routes/wasteManagementRoute.js`
- **Models**: Defined in `../prisma/schema.prisma`
