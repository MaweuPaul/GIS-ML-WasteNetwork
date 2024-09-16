import { useNavigate } from 'react-router-dom';
import React from 'react';

const Navbar = ({ activePage, setActivePage }) => {
  const navigate = useNavigate();

  const handleNavigation = (page) => {
    setActivePage(page);
    navigate(page === 'upload' ? '/upload' : '/results');
  };

  return (
    <nav className="bg-blue-700 text-white p-4 fixed top-0 left-0 right-0 z-10">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <h1 className="text-2xl font-bold">Geo Data Manager</h1>
        <div>
          <button
            onClick={() => handleNavigation('upload')}
            className={`mr-4 ${
              activePage === 'upload' ? 'font-bold border-b-2 border-white' : ''
            }`}
          >
            Upload
          </button>
          <button
            onClick={() => handleNavigation('results')}
            className={
              activePage === 'results'
                ? 'font-bold border-b-2 border-white'
                : ''
            }
          >
            Results
          </button>
        </div>
      </div>
    </nav>
  );
};
export default Navbar;
