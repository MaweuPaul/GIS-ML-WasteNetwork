const { fetchData } = require('../dataService/dataservice');

const getData = async (req, res) => {
  const { page = 1, limit = 100 } = req.query;
  try {
    const data = await fetchData(Number(page), Number(limit));
    res.json(data);
  } catch (error) {
    console.error('Error in getData:', error);
    res.status(500).json({ message: 'Failed to fetch data' });
  }
};

module.exports = {
  getData,
};
