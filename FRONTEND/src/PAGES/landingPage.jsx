import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center">
      <header className="bg-indigo-600 w-full py-4">
        <h1 className="text-white text-center text-3xl font-bold">
          Data Upload Portal
        </h1>
      </header>
      <main className="flex-grow flex flex-col items-center justify-center">
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <h1 className="text-5xl font-bold mb-4">
            Welcome to the Data Upload Portal
          </h1>
          <p className="text-lg mb-8">
            This project is designed to help you upload and manage various types
            of geographical data, including soils, protected areas, rivers,
            roads, digital elevation models, and geologies. Our goal is to
            provide a user-friendly interface for data management and
            visualization.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="py-2 px-4 bg-indigo-600 text-white font-semibold rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Get Started
          </button>
        </motion.div>
      </main>
      <footer className="w-full py-4 text-center bg-gray-200">
        <p className="text-gray-700">
          &copy; 2023 Data Upload Portal. All rights reserved.
        </p>
      </footer>
    </div>
  );
};

export default LandingPage;
