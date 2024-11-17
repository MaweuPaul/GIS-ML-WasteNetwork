const express = require('express');
const multer = require('multer');
const {
  getProtectedAreas,
  getProtectedArea,
  createProtectedArea,
  updateProtectedArea,
  deleteProtectedArea,
  deleteAllProtectedAreas,
} = require('../Controllers/protectedAreasController');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.get('/', getProtectedAreas);
router.get('/:id', getProtectedArea);
router.post('/', upload.single('file'), createProtectedArea);
router.put('/:id', updateProtectedArea);
router.delete('/deleteAll', deleteAllProtectedAreas);
router.delete('/:id', deleteProtectedArea);

module.exports = router;
