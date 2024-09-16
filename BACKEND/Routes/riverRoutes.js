const express = require('express');
const {
  getRivers,
  getRiver,
  createRiver,
  updateRiver,
  deleteRiver,
} = require('../Controllers/riverControllers');
const multer = require('multer');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.get('/', getRivers);
router.get('/:id', getRiver);
router.post('/', upload.single('file'), createRiver);
router.put('/:id', updateRiver);
router.delete('/:id', deleteRiver);

module.exports = router;
