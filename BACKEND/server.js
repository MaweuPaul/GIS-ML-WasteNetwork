const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const soilRoutes = require('./Routes/soilRoutes.js');
const protectedAreaRoutes = require('./Routes/protectedAreasRoutes.js');
const riverRoutes = require('./Routes/riverRoutes.js');
const roadRoutes = require('./Routes/roadRoutes.js');
const digitalElevationModelRoutes = require('./Routes/DemRoute.js');
const geologyRoutes = require('./Routes/geologyRoute.js');
const areaOfInterestRoutes = require('./Routes/aoiroute.js');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors({ origin: process.env.CLIENT_URL, credentials: true }));
app.use(express.json());

app.use('/soils', soilRoutes);
app.use('/protected-areas', protectedAreaRoutes);
app.use('/rivers', riverRoutes);
app.use('/roads', roadRoutes);
app.use('/digital-elevation-models', digitalElevationModelRoutes);
app.use('/geologies', geologyRoutes);

app.get('/', (req, res) => {
  res.send('Server is running');
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
