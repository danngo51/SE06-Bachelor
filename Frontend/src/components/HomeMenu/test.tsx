import React, { useState, useEffect } from "react";
import axios from "axios";

const TestBackend = () => {
  const [data, setData] = useState<any>(null); // State to store the backend response
  const [error, setError] = useState<string | null>(null); // State to store any errors

  useEffect(() => {
    // Make a GET request to the FastAPI backend endpoint
    const apiUrl = import.meta.env.VITE_BACKEND_API_URL; // This is set from .env

    axios
      .get(`${apiUrl}/api/test`)
      .then((response) => {
        setData(response.data); // Store the response data
      })
      .catch((error) => {
        setError("There was an error fetching data: " + error.message);
      });
  }, []);

  return (
    <div>
      <h1>Backend Test</h1>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {data ? (
        <pre>{JSON.stringify(data, null, 2)}</pre>
      ) : (
        <p>Loading backend data...</p>
      )}
    </div>
  );
};

export default TestBackend;
