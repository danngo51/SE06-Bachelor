import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Subdivide from './pages/Subdivide/Subdivide';
import Home from './pages/Home/Home';
import 'primereact/resources/themes/lara-light-blue/theme.css'; // Theme
import 'primereact/resources/primereact.min.css'; // Core CSS
import 'primeicons/primeicons.css'; // Icons
import 'primeflex/primeflex.css'; // Utility CSS (optional)
import { ToastContainer } from 'react-toastify'; // Import ToastContainer

function App() {
    return (
        <>
            <Router>
                <Routes>
                    <Route path="/" element={<Home />}></Route>
                    <Route path="/Map" element={<Subdivide />}></Route>
                </Routes>
            </Router>
            <ToastContainer />{' '}
            {/* Place it outside <Routes> so it's globally available */}
        </>
    );
}

export default App;
