import { useNavigate } from 'react-router-dom';
import React from 'react';

const Navbar = ({ activePage, setActivePage }) => {
  const navigate = useNavigate();

  const navItems = [
    { id: 'upload', label: 'Upload', path: '/upload', group: 'main' },
    { id: 'results', label: 'Results', path: '/results', group: 'main' },
    {
      id: 'dashboard',
      label: 'Dashboard',
      path: '/dashboard',
      icon: '📊',
      group: 'management',
    },
    {
      id: 'incidents',
      label: 'Incidents',
      path: '/incidents',
      icon: '🚨',
      group: 'management',
    },
    {
      id: 'special-pickup',
      label: 'Special Pickup',
      path: '/special-pickup',
      icon: '🚛',
      group: 'management',
    },
    {
      id: 'collection-schedule',
      label: 'Collection Schedule',
      path: '/collection-schedule',
      icon: '📆',
      group: 'management',
    },
    // {
    //   id: 'reports',
    //   label: 'Reports',
    //   path: '/reports',
    //   icon: '📋',
    //   group: 'management',
    // },
  ];

  const handleNavigation = (item) => {
    setActivePage(item.id);
    navigate(item.path);
  };

  const renderNavButton = (item) => (
    <button
      key={item.id}
      onClick={() => handleNavigation(item)}
      className={`flex items-center px-3 py-2 rounded-md transition-colors
        ${
          activePage === item.id
            ? 'bg-blue-800 text-white'
            : 'text-blue-100 hover:bg-blue-600'
        }`}
    >
      {item.icon && <span className="mr-2">{item.icon}</span>}
      {item.label}
    </button>
  );

  return (
    <nav className="bg-blue-700 text-white p-4 fixed top-0 left-0 right-0 z-10 shadow-lg">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <h1
          onClick={() => handleNavigation({ id: 'dashboard', path: '/' })}
          className="text-2xl font-bold cursor-pointer hover:text-blue-200 transition-colors"
        >
          Geo Data Manager
        </h1>

        <div className="flex items-center space-x-6">
          {/* Main Group */}
          <div className="flex items-center space-x-2">
            {navItems
              .filter((item) => item.group === 'main')
              .map((item) => renderNavButton(item))}
          </div>

          {/* Divider */}
          <div className="h-6 w-px bg-blue-400"></div>

          {/* Management Group */}
          <div className="flex items-center space-x-2">
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
