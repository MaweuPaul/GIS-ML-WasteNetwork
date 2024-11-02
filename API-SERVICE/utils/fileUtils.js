const fs = require('fs').promises;
const path = require('path');

const fileUtils = {
  // Delete a single file
  deleteFile: async (filePath) => {
    try {
      await fs.unlink(filePath);
      console.log(`Successfully deleted file: ${filePath}`);
    } catch (error) {
      if (error.code !== 'ENOENT') {
        // Ignore if file doesn't exist
        console.error(`Error deleting file ${filePath}:`, error);
        throw error;
      }
    }
  },

  // Delete multiple files
  deleteFiles: async (filePaths) => {
    if (!Array.isArray(filePaths)) return;

    const deletePromises = filePaths.map((filePath) =>
      fileUtils.deleteFile(filePath)
    );

    try {
      await Promise.allSettled(deletePromises);
    } catch (error) {
      console.error('Error deleting multiple files:', error);
      throw error;
    }
  },
};

module.exports = fileUtils;
