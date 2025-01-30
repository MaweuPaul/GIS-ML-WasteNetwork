const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const path = require('path');
const soilRoutes = require('./Routes/soilRoutes.js');
const protectedAreaRoutes = require('./Routes/protectedAreasRoutes.js');
const riverRoutes = require('./Routes/riverRoutes.js');
const roadRoutes = require('./Routes/roadRoutes.js');
const digitalElevationModelRoutes = require('./Routes/demRoute.js');
const areaOfInterestRoutes = require('./Routes/aoiroute.js');
const landuseRasterRoute = require('./Routes/landuseRoute.js');
const settlementRoutes = require('./Routes/settlementRoute.js');
const geologyRoutes = require('./Routes/geologyRoute.js');
const databaseRoutes = require('./Routes/databaseRoutes.js');
const dataRoutes = require('./Routes/dataRoutes.js');
const specialPickupRoutes = require('./Routes/specialPickupRoutes.js');
const reportIncidence = require('./Routes/reportIssueRoute.js');
const zones = require('./Routes/zoneRoute.js');
const wasteManagementRoutes = require('./Routes/wasteManagementRoute.js');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Allow multiple origins
const allowedOrigins = [
  'http://localhost:5173',
  'http://localhost:5174',
  process.env.CLIENT_URL,
].filter(Boolean);

app.use(
  cors({
    origin: function (origin, callback) {
      // Allow requests with no origin (like mobile apps or curl requests)
      if (!origin) return callback(null, true);

      if (allowedOrigins.indexOf(origin) !== -1) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
  })
);

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(
  '/uploads',
  (req, res, next) => {
    next();
  },
  express.static(path.join(__dirname, 'uploads'), {
    fallthrough: true,
    maxAge: '1d',
  })
);
// response headers for better security
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  next();
});
app.use(express.urlencoded({ extended: true }));

app.use('/api/soils', soilRoutes);
app.use('/api/protected-areas', protectedAreaRoutes);
app.use('/api/rivers', riverRoutes);
app.use('/api/roads', roadRoutes);
app.use('/api/digital-elevation-models', digitalElevationModelRoutes);
app.use('/api/area-of-interest', areaOfInterestRoutes);
app.use('/api/land-use-raster', landuseRasterRoute);
app.use('/api/settlement', settlementRoutes);
app.use('/api/database', databaseRoutes);
app.use('/api/geology', geologyRoutes);
app.use('/api/data', dataRoutes);
app.use('/api/specialPickup', specialPickupRoutes);
app.use('/api/incidents', reportIncidence);
app.use('/api/zones', zones);

app.use('/api/waste-management', wasteManagementRoutes);

// Catch-all route for API
app.use('/api/*', (req, res) => {
  res.status(404).json({ message: 'API route not found' });
});

app.get('/', (req, res) => {
  res.send('Server is running');
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
