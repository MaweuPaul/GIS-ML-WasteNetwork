import { useNavigate } from 'react-router-dom';
import React from 'react';

const Navbar = ({ activePage, setActivePage, handleCleanDatabase }) => {
  const navigate = useNavigate();

  const handleNavigation = (page) => {
    setActivePage(page);
    switch (page) {
      case 'upload':
        navigate('/upload');
        break;
      case 'results':
        navigate('/results');
        break;
      case 'incidents':
        navigate('/incidents');
        break;
      case 'dashboard':
        navigate('/dashboard');
        break;
      case 'reports':
        navigate('/reports');
      case 'special-pickup':
        navigate('/special-pickup');
        break;
      default:
        navigate('/');
    }
  };

  return (
    <nav className="bg-blue-700 text-white p-4 fixed top-0 left-0 right-0 z-10">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <h1 className="text-2xl font-bold">Geo Data Manager</h1>

        <div className="flex items-center space-x-6">
          {/* Original Buttons */}
          <div className="flex items-center">
            <button
              onClick={handleCleanDatabase}
              className="mr-5 py-1 px-4 bg-red-600 text-white rounded hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50"
            >
              Clean Database
            </button>
            <button
              onClick={() => handleNavigation('upload')}
              className={`mr-4 ${
                activePage === 'upload'
                  ? 'font-bold border-b-2 border-white'
                  : ''
              }`}
            >
              Upload
            </button>
            <button
              onClick={() => handleNavigation('results')}
              className={`mr-4 ${
                activePage === 'results'
                  ? 'font-bold border-b-2 border-white'
                  : ''
              }`}
            >
              Results
            </button>
          </div>

          {/* Divider */}
          <div className="h-6 w-px bg-blue-400"></div>

          {/* Incident Management Buttons */}
          <div className="flex items-center space-x-4">
            <button
              onClick={() => handleNavigation('dashboard')}
              className={`flex items-center ${
                activePage === 'dashboard'
                  ? 'font-bold border-b-2 border-white'
                  : ''
              }`}
            >
              <span className="mr-1">📊</span>
              Dashboard
            </button>

            <button
              onClick={() => handleNavigation('incidents')}
              className={`flex items-center ${
                activePage === 'incidents'
                  ? 'font-bold border-b-2 border-white'
                  : ''
              }`}
            >
              <span className="mr-1">🚨</span>
              Incidents
            </button>
            <button
              onClick={() => handleNavigation('special-pickup')}
              className={`flex items-center ${
                activePage === 'special-pickup'
                  ? 'font-bold border-b-2 border-white'
                  : ''
              }`}
            >
              <span className="mr-1">🚛</span>
              Special Pickup
            </button>
            <button
              onClick={() => handleNavigation('reports')}
              className={`flex items-center ${
                activePage === 'reports'
                  ? 'font-bold border-b-2 border-white'
                  : ''
              }`}
            >
              <span className="mr-1">📋</span>
              Reports
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
