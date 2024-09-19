const express = require('express');
const multer = require('multer');
const path = require('path');
const areaOfInterestController = require('../Controllers/aoicontroller.js');

const router = express.Router();

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, 'area_of_interest' + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (path.extname(file.originalname).toLowerCase() === '.shp') {
      cb(null, true);
    } else {
      cb(new Error('Only .shp files are allowed.'));
    }
  },
});

router.post(
  '/upload',
  upload.single('file'),
  areaOfInterestController.createAreaOfInterest
);

module.exports = router;
