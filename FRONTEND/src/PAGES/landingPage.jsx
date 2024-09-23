import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { FaUpload, FaMap, FaChartLine, FaCheckCircle } from 'react-icons/fa';

const LandingPage = () => {
  const navigate = useNavigate();
  const currentYear = new Date().getFullYear();

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-green-600 w-full py-6 shadow-md">
        <h1 className="text-white text-center text-4xl font-bold">
          Waste Disposal Suitability Analysis
        </h1>
      </header>
      <main className="flex-grow flex flex-col items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center max-w-3xl"
        >
          <h1 className="text-5xl font-bold mb-6">
            Welcome to the Suitability Analysis Portal
          </h1>
          <p className="text-lg mb-8 text-gray-700">
            This project leverages machine learning to analyze the suitability
            of various locations for waste disposal and collection. Upload and
            manage geographical data such as soils, protected areas, rivers,
            roads, digital elevation models, and geologies to get started.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="py-3 px-6 bg-green-600 text-white font-semibold rounded-md shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition duration-300"
          >
            Get Started
          </button>
        </motion.div>
        <section className="mt-16 text-center">
          <h2 className="text-3xl font-bold mb-6">Features</h2>
          <ul className="text-lg mb-12 space-y-4 text-gray-700">
            <li>Upload and manage various geographical data types</li>
            <li>Visualize data on an interactive map</li>
            <li>
              Analyze suitability for waste disposal using machine learning
            </li>
          </ul>
        </section>
        <section className="mt-16 text-center">
          <h2 className="text-3xl font-bold mb-6">How It Works</h2>
          <p className="text-lg mb-12 text-gray-700">
            Upload your geographical data, and our machine learning algorithms
            will analyze the suitability of different locations for waste
            disposal and collection. Visualize the results on an interactive
            map.
          </p>
        </section>
        <section className="mt-16 text-center">
          <h2 className="text-3xl font-bold mb-6">Procedure</h2>
          <div className="flex justify-center items-center space-x-12">
            <div className="flex flex-col items-center">
              <FaUpload className="text-6xl text-green-600 mb-2" />
              <p className="text-lg text-gray-700">1. Upload Data</p>
            </div>
            <div className="flex flex-col items-center">
              <FaMap className="text-6xl text-green-600 mb-2" />
              <p className="text-lg text-gray-700">2. Visualize Data</p>
            </div>
            <div className="flex flex-col items-center">
              <FaChartLine className="text-6xl text-green-600 mb-2" />
              <p className="text-lg text-gray-700">3. Analyze Suitability</p>
            </div>
            <div className="flex flex-col items-center">
              <FaCheckCircle className="text-6xl text-green-600 mb-2" />
              <p className="text-lg text-gray-700">4. Get Results</p>
            </div>
          </div>
        </section>
      </main>
      <footer className="w-full py-6 text-center bg-gray-200 mt-16">
        <p className="text-gray-700">
          &copy; {currentYear} Waste Disposal Suitability Analysis. All rights
          reserved.
        </p>
      </footer>
    </div>
  );
};

export default LandingPage;
