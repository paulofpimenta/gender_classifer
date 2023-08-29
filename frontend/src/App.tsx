import './App.css';
import { useState, useEffect } from "react";
import axios from "axios";
import WebCam from './components/WebCam';
import { BrowserRouter as Router, Routes, Route }
    from "react-router-dom";
import Main from './Main';
import Home from './components/Home';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/webcam" element={<WebCam />}/>
        <Route path="/" element={<Home/>}/>
      </Routes>
    </Router>
  );
}
  
export default App;

