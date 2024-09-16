const fs = require('fs');
const path = require('path');

exports.uploadAreaOfInterest = (req, res) => {
  if (!req.file) {
    return res.status(400).send('No file uploaded.');
  }

  const fileExtension = path.extname(req.file.originalname).toLowerCase();
  if (fileExtension !== '.shp') {
    fs.unlinkSync(req.file.path); // Delete the uploaded file
    return res.status(400).send('Only .shp files are allowed.');
  }

  // Here you would typically process the shapefile, perhaps saving its path to a database
  // For now, we'll just send a success response
  res.status(200).send('Area of Interest shapefile uploaded successfully.');
};
