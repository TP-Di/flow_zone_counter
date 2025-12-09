import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import CameraManager from './pages/CameraManager';
import ZoneDrawer from './pages/ZoneDrawer';
import AnnotationTool from './pages/AnnotationTool';
import Dashboard from './pages/Dashboard';

function Navigation() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Camera Management', icon: '📷' },
    { path: '/zones', label: 'Zone Configuration', icon: '🎯' },
    { path: '/annotate', label: 'Annotation Tool', icon: '✏️' },
    { path: '/dashboard', label: 'Dashboard', icon: '📊' },
  ];

  return (
    <nav className="bg-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex space-x-8">
            <div className="flex items-center">
              <span className="text-2xl font-bold text-primary-600">People Counter</span>
            </div>
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                  location.pathname === item.path
                    ? 'border-primary-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<CameraManager />} />
            <Route path="/zones" element={<ZoneDrawer />} />
            <Route path="/annotate" element={<AnnotationTool />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
