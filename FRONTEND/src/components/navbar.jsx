// components/Navbar.jsx
import { useNavigate } from 'react-router-dom';
import React from 'react';

const Navbar = ({ activePage, setActivePage, handleCleanDatabase }) => {
  const navigate = useNavigate();

  const navItems = [
    { id: 'upload', label: 'Upload', icon: null, group: 'main' },
    { id: 'results', label: 'Results', icon: null, group: 'main' },
    { id: 'dashboard', label: 'Dashboard', icon: '📊', group: 'management' },
    { id: 'incidents', label: 'Incidents', icon: '🚨', group: 'management' },
    {
      id: 'special-pickup',
      label: 'Special Pickup',
      icon: '🚛',
      group: 'management',
    },
    {
      id: 'collection-schedule',
      label: 'Collection Schedule',
      icon: '📆',
      group: 'management',
    },
    { id: 'reports', label: 'Reports', icon: '📋', group: 'management' },
  ];

  const handleNavigation = (page) => {
    setActivePage(page);
    // Use template literals for better readability
    navigate(`/${page === 'dashboard' ? '' : page}`);
  };

  const renderNavButton = (item) => (
    <button
      key={item.id}
      onClick={() => handleNavigation(item.id)}
      className={`flex items-center ${
        activePage === item.id ? 'font-bold border-b-2 border-white' : ''
      }`}
    >
      {item.icon && <span className="mr-1">{item.icon}</span>}
      {item.label}
    </button>
  );

  return (
    <nav className="bg-blue-700 text-white p-4 fixed top-0 left-0 right-0 z-10">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <h1 className="text-2xl font-bold">Geo Data Manager</h1>

        <div className="flex items-center space-x-6">
          {/* Main Group */}
          <div className="flex items-center">
            <button
              onClick={handleCleanDatabase}
              className="mr-5 py-1 px-4 bg-red-600 text-white rounded hover:bg-red-700 
                focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50"
            >
              Clean Database
            </button>
            {navItems
              .filter((item) => item.group === 'main')
              .map((item) => (
                <div key={item.id} className="mr-4">
                  {renderNavButton(item)}
                </div>
              ))}
          </div>

          {/* Divider */}
          <div className="h-6 w-px bg-blue-400"></div>

          {/* Management Group */}
          <div className="flex items-center space-x-4">
            {navItems
              .filter((item) => item.group === 'management')
              .map(renderNavButton)}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
