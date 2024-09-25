const express = require('express');
const { getData } = require('../Controllers/dataServiceController');

const router = express.Router();

router.get('/', getData);

module.exports = router;
