import React, { useState, useEffect } from 'react';
import { collection, getDocs, query, where, orderBy } from "firebase/firestore";
import { db } from "../firebase/config";
import { useAuth } from "../firebase/AuthContext";
import "./SustainabilityReport.css";

const SustainabilityReport = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const { currentUser } = useAuth();

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        // Only fetch records for the current user
        const q = query(
          collection(db, "emission_records"),
          where("userId", "==", currentUser.uid),
          orderBy("createdAt", "desc")
        );
        
        const snapshot = await getDocs(q);
        const records = snapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data(),
          createdAt: doc.data().createdAt?.toDate()
        }));

        setHistory(records);

        // Calculate statistics
        if (records.length > 0) {
          const totalEmissions = records.reduce((sum, record) => sum + record.emission, 0);
          const avgEmission = totalEmissions / records.length;
          const sortedEmissions = records.map(r => r.emission).sort((a, b) => a - b);
          
          setStats({
            totalRecords: records.length,
            totalEmissions,
            avgEmission,
            minEmission: sortedEmissions[0],
            maxEmission: sortedEmissions[sortedEmissions.length - 1],
            latestEmission: records[0].emission
          });
        }
      } catch (err) {
        console.error("Error fetching history:", err);
        setError("Failed to load emission history");
      } finally {
        setLoading(false);
      }
    };

    if (currentUser) {
      fetchHistory();
    }
  }, [currentUser]);

  if (loading) return <div className="loading">Loading sustainability report...</div>;
  if (error) return <div className="error">{error}</div>;
  if (!history.length) return <div className="no-data">No emission records found</div>;

  return (
    <div className="sustainability-report">
      <h3>🌍 Your Sustainability Report</h3>
      
      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <h4>Total Records</h4>
            <p>{stats.totalRecords}</p>
          </div>
          <div className="stat-card">
            <h4>Total Emissions</h4>
            <p>{stats.totalEmissions.toFixed(2)} kg CO₂</p>
          </div>
          <div className="stat-card">
            <h4>Average Emission</h4>
            <p>{stats.avgEmission.toFixed(2)} kg CO₂</p>
          </div>
          <div className="stat-card">
            <h4>Lowest Emission</h4>
            <p>{stats.minEmission.toFixed(2)} kg CO₂</p>
          </div>
          <div className="stat-card">
            <h4>Highest Emission</h4>
            <p>{stats.maxEmission.toFixed(2)} kg CO₂</p>
          </div>
          <div className="stat-card highlight">
            <h4>Latest Emission</h4>
            <p>{stats.latestEmission.toFixed(2)} kg CO₂</p>
          </div>
        </div>
      )}

      <div className="history-table">
        <h4>Emission History</h4>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Distance (km)</th>
                <th>Transport</th>
                <th>Diet</th>
                <th>Grocery Bill</th>
                <th>Screen Time</th>
                <th>Emission (kg CO₂)</th>
              </tr>
            </thead>
            <tbody>
              {history.map((entry) => (
                <tr key={entry.id}>
                  <td>{entry.createdAt?.toLocaleDateString() || "Unknown"}</td>
                  <td>{entry.distance}</td>
                  <td>{entry.transport_type}</td>
                  <td>{entry.diet}</td>
                  <td>{entry.grocery}</td>
                  <td>{entry.screen_time}</td>
                  <td className="emission-cell">{entry.emission.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default SustainabilityReport;
