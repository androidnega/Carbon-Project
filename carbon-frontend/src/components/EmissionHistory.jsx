import { useState, useEffect } from 'react';
import { collection, getDocs, query, orderBy, where } from "firebase/firestore";
import { db } from "../firebase/config";
import { useAuth } from '../firebase/AuthContext';
import './EmissionHistory.css';

const EmissionHistory = () => {
    const [history, setHistory] = useState([]);
    const { currentUser } = useAuth();

    useEffect(() => {
        const fetchHistory = async () => {
            if (!currentUser) return;
            
            // Only fetch records for the current user
            const q = query(
                collection(db, "emission_records"),
                where("userId", "==", currentUser.uid),
                orderBy("createdAt", "desc")
            );
            
            try {
                const snapshot = await getDocs(q);
                const records = snapshot.docs.map(doc => ({
                    id: doc.id,
                    ...doc.data()
                }));
                setHistory(records);
            } catch (error) {
                console.error("Error fetching emission history:", error);
            }
        };

        fetchHistory();
    }, [currentUser]);

    if (history.length === 0) {
        return (
            <div className="emission-history-empty">
                <h3>🌍 Sustainability Report</h3>
                <p>No emission records found. Start tracking your carbon footprint!</p>
            </div>
        );
    }

    return (
        <div className="emission-history">
            <h3>🌍 Sustainability Report</h3>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Distance</th>
                        <th>Diet</th>
                        <th>Emission (kg)</th>
                    </tr>
                </thead>
                <tbody>
                    {history.map((entry) => (
                        <tr key={entry.id}>
                            <td>{entry.createdAt?.toDate?.().toLocaleString() || "Unknown"}</td>
                            <td>{entry.distance} km</td>
                            <td>{entry.diet}</td>
                            <td className="emission-value">{entry.emission}</td>
                        </tr>
                    ))}
                </tbody>
                <tfoot>
                    <tr>
                        <td colSpan="3">Total Emissions:</td>
                        <td className="emission-value">
                            {history.reduce((sum, entry) => sum + Number(entry.emission), 0).toFixed(2)} kg
                        </td>
                    </tr>
                </tfoot>
            </table>
        </div>
    );
};

export default EmissionHistory;
