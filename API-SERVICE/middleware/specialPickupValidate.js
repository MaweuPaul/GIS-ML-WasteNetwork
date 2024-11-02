const validateSpecialPickup = (req, res, next) => {
  const {
    type,
    quantity,
    description,
    preferredDate,
    preferredTime,
    location,
    contactName,
    contactPhone,
    contactEmail,
  } = req.body;

  // Basic validation
  if (
    !type ||
    !['GENERAL', 'RECYCLABLE', 'ORGANIC', 'HAZARDOUS', 'E_WASTE'].includes(type)
  ) {
    return res.status(400).json({ error: 'Invalid waste type' });
  }

  if (!quantity || !description) {
    return res
      .status(400)
      .json({ error: 'Quantity and description are required' });
  }

  if (!preferredDate || !preferredTime) {
    return res
      .status(400)
      .json({ error: 'Preferred date and time are required' });
  }

  if (!location || !location.lat || !location.lng) {
    return res.status(400).json({ error: 'Valid location is required' });
  }

  if (!contactName || !contactPhone || !contactEmail) {
    return res.status(400).json({ error: 'Contact information is required' });
  }

  // Validate email format
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(contactEmail)) {
    return res.status(400).json({ error: 'Invalid email format' });
  }

  // Validate phone format (basic)
  const phoneRegex = /^\+?[\d\s-]{10,}$/;
  if (!phoneRegex.test(contactPhone)) {
    return res.status(400).json({ error: 'Invalid phone number format' });
  }

  next();
};

module.exports = { validateSpecialPickup };
