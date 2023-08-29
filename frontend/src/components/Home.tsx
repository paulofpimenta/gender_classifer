import './css/Home.css';
import { useState, useEffect } from "react";
import axios from "axios";
import { Link } from 'react-router-dom';


type greeting = {
  Hello: string;
};

export default function Home() {
  const [loading, setLoading] = useState(true) // Loading state
  const [result, setResult] = useState<greeting>();

  useEffect(() => {
    const api = async () => {
      const data = await axios.get("http://localhost:8000", {
        method: "GET"
      });
      setResult(data.data);
      setLoading(false) // Setting the loading state to false after data is set.
    };
    api();
  }, []);

  return (
    <div className="App">
      <h1>
       {/* Checking for loading state before rendering the data */}
      {loading ? (
         <p>Loading... </p>
      ) : (
        `Hello ` + result?.Hello  + ` from back`         
      )}
      </h1>
      <h2><Link to='/webcam'>Image from WebCam</Link></h2>
    </div>
  );
}

