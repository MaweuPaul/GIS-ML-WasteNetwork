const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const soilRoutes = require('./Routes/soilRoutes.js');
const protectedAreaRoutes = require('./Routes/protectedAreasRoutes.js');
const riverRoutes = require('./Routes/riverRoutes.js');
const roadRoutes = require('./Routes/roadRoutes.js');
// const digitalElevationModelRoutes = require('./Routes/demRoute.js');
// const geologyRoutes = require('./Routes/geologyRoute.js');
// const areaOfInterestRoutes = require('./Routes/aoiroute.js');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors({ origin: process.env.CLIENT_URL, credentials: true }));
app.use(express.json());

app.use('/api/soils', soilRoutes);
app.use('/api/protected-areas', protectedAreaRoutes);
app.use('/api/rivers', riverRoutes);
app.use('/api/roads', roadRoutes);
// app.use('/api/digital-elevation-models', digitalElevationModelRoutes);
// app.use('/api/geologies', geologyRoutes);
// app.use('/api/area-of-interest', areaOfInterestRoutes);

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
