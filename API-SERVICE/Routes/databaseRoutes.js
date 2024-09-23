const express = require('express');
const { checkDatabaseEmpty } = require('../Controllers/databaseController');

const router = express.Router();

router.get('/check', checkDatabaseEmpty);

module.exports = router;
