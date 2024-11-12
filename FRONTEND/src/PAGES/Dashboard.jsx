import React, { useEffect, useState } from 'react';
import {
  FaTrash,
  FaRoute,
  FaCalendarAlt,
  FaExclamationTriangle,
  FaChartBar,
  FaMapMarkedAlt,
  FaTruck,
  FaClipboardList,
} from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const API_URL = 'http://localhost:3000';

const Dashboard = () => {
  const navigate = useNavigate();
  const [summaryData, setSummaryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
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
      icon: <FaTrash className="text-blue-500" />,
      value: summaryData ? `${summaryData.landfillCount} Sites` : 'Loading...',
      description: 'View and manage landfill sites',
      path: '/results',
      color: 'bg-blue-100',
    },
    {
      title: 'Collection Points',
      icon: <FaRoute className="text-green-500" />,
      value: summaryData
        ? `${summaryData.collectionPointCount} Points`
        : 'Loading...',
      description: 'View collection points',
      path: '/collection-schedule',
      color: 'bg-green-100',
    },
    {
      title: 'Active Routes',
      icon: <FaExclamationTriangle className="text-red-500" />,
      value: summaryData ? `${summaryData.routeCount} Routes` : 'Loading...',
      description: 'View active routes',
      path: '/incidents',
      color: 'bg-red-100',
    },
    {
      title: 'Average Distance',
      icon: <FaTruck className="text-purple-500" />,
      value: summaryData
        ? `${(summaryData.routeStats.averageDistance / 1000).toFixed(2)} km`
        : 'Loading...',
      description: 'Average route distance',
      path: '/special-pickup',
      color: 'bg-purple-100',
    },
  ];

  const quickActions = [
    {
      title: 'Upload Data',
      icon: <FaMapMarkedAlt className="text-indigo-500" />,
      description: 'Upload new geographical data',
      path: '/upload',
    },
    {
      title: 'View Reports',
      icon: <FaClipboardList className="text-orange-500" />,
      description: 'Access analytics and reports',
      path: '/reports',
    },
  ];

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center">
        <div className="text-red-500 mb-4">Error: {error}</div>
        <button
          onClick={() => window.location.reload()}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Welcome to the Waste Management System
        </p>
      </div>

      {/* Main Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {cards.map((card, index) => (
          <div
            key={index}
            onClick={() => navigate(card.path)}
            className={`${card.color} p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer`}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="text-2xl">{card.icon}</div>
              <span className="text-sm font-semibold text-gray-600">
                {card.value}
              </span>
            </div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              {card.title}
            </h3>
            <p className="text-sm text-gray-600">{card.description}</p>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {quickActions.map((action, index) => (
            <div
              key={index}
              onClick={() => navigate(action.path)}
              className="bg-white p-4 rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer flex items-center space-x-4"
            >
              <div className="text-2xl">{action.icon}</div>
              <div>
                <h3 className="font-semibold text-gray-800">{action.title}</h3>
                <p className="text-sm text-gray-600">{action.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary Section */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Route Statistics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border-r border-gray-200 pr-4">
            <h3 className="text-sm font-medium text-gray-600">Min Distance</h3>
            <p className="text-2xl font-semibold text-gray-800">
              {(summaryData.routeStats.minDistance / 1000).toFixed(2)} km
            </p>
          </div>
          <div className="border-r border-gray-200 px-4">
            <h3 className="text-sm font-medium text-gray-600">Max Distance</h3>
            <p className="text-2xl font-semibold text-gray-800">
              {(summaryData.routeStats.maxDistance / 1000).toFixed(2)} km
            </p>
          </div>
          <div className="pl-4">
            <h3 className="text-sm font-medium text-gray-600">Total Routes</h3>
            <p className="text-2xl font-semibold text-gray-800">
              {summaryData.routeCount}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
