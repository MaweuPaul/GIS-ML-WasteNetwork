const express = require('express');
const multer = require('multer');
const {
  uploadLandUseRaster,
  getLandUseRaster,
  deleteAllLanduse,
} = require('../Controllers/landuseController');

const router = express.Router();

// Configure Multer for memory storage
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'image/tiff' || file.mimetype === 'image/x-tiff') {
      cb(null, true);
    } else {
      cb(new Error('Only TIFF files are allowed'), false);
    }
  },
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  },
});

router.post('/', upload.single('file'), uploadLandUseRaster);
router.delete('/deleteAll', deleteAllLanduse);
router.get('/:id', getLandUseRaster);

module.exports = router;
