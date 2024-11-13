import React, { useEffect, useState } from 'react';
import {
  FaTrash,
  FaMapMarkerAlt,
  FaRoute,
  FaTruck,
  FaMapMarkedAlt,
  FaClipboardList,
  FaChartLine,
  FaCalendarAlt,
} from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';

const API_URL = 'http://localhost:3000';

// Stat Card Component
const StatCard = ({ title, value, icon, description }) => (
  <motion.div
    whileHover={{ y: -5 }}
    className="p-6 rounded-lg bg-gray-50 border border-gray-200 transition-all duration-300"
  >
    <div className="flex items-center justify-between mb-2">
      <div className="p-2 rounded-full bg-white shadow-sm">{icon}</div>
      <span className="text-lg font-semibold text-gray-700">{value}</span>
    </div>
    <h3 className="text-gray-800 text-lg font-bold mb-1">{title}</h3>
    <p className="text-gray-600 text-sm">{description}</p>
  </motion.div>
);

// Quick Action Card Component
const QuickActionCard = ({ title, description, icon, onClick, bgColor }) => (
  <motion.div
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    onClick={onClick}
    className={`${bgColor} p-6 rounded-xl border border-gray-200 
                transition-all duration-300 cursor-pointer flex items-center space-x-4`}
  >
    <div className="p-3 rounded-full bg-white shadow-sm text-2xl">{icon}</div>
    <div>
      <h3 className="font-bold text-gray-800 text-lg">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  </motion.div>
);

// Loading Spinner Component
const LoadingSpinner = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    className="flex items-center justify-center min-h-screen bg-gray-50"
  >
    <div className="text-center">
      <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      <p className="mt-4 text-gray-600 font-medium">
        Loading your dashboard...
      </p>
    </div>
  </motion.div>
);

const Dashboard = () => {
  const navigate = useNavigate();
  const [summaryData, setSummaryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeOfDay, setTimeOfDay] = useState('');

  useEffect(() => {
    // Set time of day greeting
    const hour = new Date().getHours();
    if (hour < 12) setTimeOfDay('morning');
    else if (hour < 17) setTimeOfDay('afternoon');
    else setTimeOfDay('evening');

    // Fetch dashboard data
    const fetchSummaryData = async () => {
      try {
        const response = await axios.get(
          `${API_URL}/api/waste-management/summary`
        );
        setSummaryData(response.data.data);
      } catch (err) {
        setError('Failed to fetch dashboard data');
        console.error('Error fetching summary:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchSummaryData();
  }, []);

  const cards = [
    {
      title: 'Landfill Sites',
      icon: <FaTrash className="text-blue-600 text-2xl" />,
      value: summaryData ? `${summaryData.landfillCount} Sites` : '...',
      description: 'View and manage landfill sites',
      path: '/landfills',
      color: 'bg-blue-50',
      borderColor: 'border-blue-200',
    },
    {
      title: 'Collection Points',
      icon: <FaMapMarkerAlt className="text-green-600 text-2xl" />,
      value: summaryData ? `${summaryData.collectionPointCount} Points` : '...',
      description: 'Monitor collection points',
      path: '/collection-points',
      color: 'bg-green-50',
      borderColor: 'border-green-200',
    },
    {
      title: 'Active Routes',
      icon: <FaRoute className="text-red-600 text-2xl" />,
      value: summaryData ? `${summaryData.routeCount} Routes` : '...',
      description: 'Track active routes',
      path: '/routes',
      color: 'bg-red-50',
      borderColor: 'border-red-200',
    },
    {
      title: 'Average Distance',
      icon: <FaTruck className="text-purple-600 text-2xl" />,
      value: summaryData
        ? `${(summaryData.routeStats?.averageDistance / 1000).toFixed(2)} km`
        : '...',
      description: 'Route statistics',
      path: '/statistics',
      color: 'bg-purple-50',
      borderColor: 'border-purple-200',
    },
  ];

  const quickActions = [
    {
      title: 'Upload Data',
      icon: <FaMapMarkedAlt className="text-indigo-600" />,
      description: 'Upload new geographical data',
      path: '/upload',
      bgColor: 'bg-indigo-50',
    },
    {
      title: 'View Reports',
      icon: <FaClipboardList className="text-orange-600" />,
      description: 'Access analytics and reports',
      path: '/reports',
      bgColor: 'bg-orange-50',
    },
  ];

  if (loading) return <LoadingSpinner />;

  if (error) {
    return (
      <div className="p-6 text-center">
        <div className="text-red-500 mb-4">Error: {error}</div>
        <button
          onClick={() => window.location.reload()}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-8 max-w-7xl mx-auto"
    >
      {/* Header Section */}
      <div className="mb-10">
        <motion.h1
          initial={{ x: -20 }}
          animate={{ x: 0 }}
          className="text-4xl font-bold text-gray-800"
        >
          Good {timeOfDay}! 👋
        </motion.h1>
        <p className="text-gray-600 mt-2 text-lg">
          Welcome to your Waste Management Dashboard
        </p>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {cards.map((card, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`${card.color} p-6 rounded-xl border ${card.borderColor} 
                       transition-all duration-300 cursor-pointer
                       transform hover:scale-105 hover:shadow-lg`}
            onClick={() => navigate(card.path)}
          >
            <StatCard
              title={card.title}
              value={card.value}
              icon={card.icon}
              description={card.description}
            />
          </motion.div>
        ))}
      </div>

      {/* Quick Actions Section */}
      <div className="mb-10">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {quickActions.map((action, index) => (
            <QuickActionCard
              key={index}
              title={action.title}
              description={action.description}
              icon={action.icon}
              onClick={() => navigate(action.path)}
              bgColor={action.bgColor}
            />
          ))}
        </div>
      </div>

      {/* Route Statistics Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-sm p-8 border border-gray-200"
      >
        <h2 className="text-2xl font-bold text-gray-800 mb-6">
          Route Statistics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <StatCard
            title="Min Distance"
            value={`${(summaryData.routeStats?.minDistance / 1000).toFixed(
              2
            )} km`}
            icon={<FaChartLine className="text-green-500" />}
            description="Shortest route distance"
          />
          <StatCard
            title="Max Distance"
            value={`${(summaryData.routeStats?.maxDistance / 1000).toFixed(
              2
            )} km`}
            icon={<FaChartLine className="text-red-500" />}
            description="Longest route distance"
          />
          <StatCard
            title="Total Routes"
            value={summaryData.routeCount}
            icon={<FaCalendarAlt className="text-blue-500" />}
            description="Active routes today"
          />
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Dashboard;
