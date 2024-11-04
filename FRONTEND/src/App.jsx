import React, { useState } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom';
import LandingPage from './PAGES/landingPage';
import UploadPage from './PAGES/uploadPage';

import Dashboard from './PAGES/Dashboard';
import Incidents from './PAGES/Incidents';
import Reports from './PAGES/Reports';
import Navbar from './components/navbar';
import Results from './PAGES/resultsPage';
import SpecialRequests from './PAGES/SpecialRequests';
import CollectionScheduleManager from './PAGES/CollectionScheduleManager';

const App = () => {
  const [activePage, setActivePage] = useState('dashboard');
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navbar activePage={activePage} setActivePage={setActivePage} />

        <div className="pt-16">
          {' '}
          {/* Add padding to account for fixed navbar */}
          <Routes>
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/results" element={<Results />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/incidents" element={<Incidents />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/special-pickup" element={<SpecialRequests />} />
            <Route
              path="/collection-schedule"
              element={<CollectionScheduleManager />}
            />
            <Route path="/" element={<Dashboard />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;
