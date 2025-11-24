import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation, useNavigate } from 'react-router-dom';
import Home from './components/Home';
import ModelSelector from './components/ModelSelector';
import Processing from './components/Processing';

const AppContent = () => {
    const location = useLocation();
    const navigate = useNavigate();

    useEffect(() => {
        const hasVisited = sessionStorage.getItem('app_visited');
        if (!hasVisited) {
            if (location.pathname === '/') {
                sessionStorage.setItem('app_visited', 'true');
            } else {
                navigate('/');
            }
        }
    }, [location.pathname, navigate]);

    useEffect(() => {
        const handleBeforeUnload = () => {
            sessionStorage.removeItem('app_visited');
        };
        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, []);


    return (
        <div className="flex flex-col min-h-screen bg-gray-50">
            <nav className="bg-white text-gray-800 shadow-md sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        <Link to="/" className="font-bold text-xl hover:text-blue-600 transition-colors">
                            FaceBack SimSnap
                        </Link>
                    </div>
                </div>
            </nav>
            
            <main className="flex-grow">
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/select-model/:jobId" element={<ModelSelector />} />
                    <Route path="/processing/:jobId" element={<Processing />} />
                </Routes>
            </main>

            <footer className="bg-white text-center text-sm text-gray-500 p-4 mt-8 border-t">
                © 2024 FaceBack SimSnap. All Rights Reserved.
            </footer>
        </div>
    );
};


function App() {
    return (
        <Router>
            <AppContent />
        </Router>
    );
}

export default App;