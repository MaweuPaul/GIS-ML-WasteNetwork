const express = require('express');
const router = express.Router();
const settlementController = require('../Controllers/setllementController');

// GET all settlements
router.get('/', settlementController.getSettlements);

// GET a single settlement by ID
router.get('/:id', settlementController.getSettlement);

// POST create new settlements (handles chunked data)
router.post('/', settlementController.createSettlements);

// PUT update a settlement
router.put('/:id', settlementController.updateSettlement);

router.delete('/deleteAll', settlementController.deleteAllSettlements);
// DELETE a settlement
router.delete('/:id', settlementController.deleteSettlement);

// DELETE all settlements

module.exports = router;
