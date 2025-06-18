import React, { useState } from "react";
import axios from "axios";
import { db } from "../firebase/config";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { useAuth } from "../firebase/AuthContext";
import "./CarbonForm.css";

const CarbonForm = () => {
  const { currentUser } = useAuth();
  const [form, setForm] = useState({
    distance: "",
    transport_type: "",
    body_type: "",
    sex: "",
    diet: "",
    grocery: "",
    screen_time: "",
  });

  const [emission, setEmission] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [saveStatus, setSaveStatus] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSaveStatus(null);

    try {
      // First, get the prediction from the API
      const res = await axios.post("http://localhost:5000/api/predict", form);
      const predictionResult = res.data.emission;
      setEmission(predictionResult);

      // Then, save the result to Firestore
      await addDoc(collection(db, "emission_records"), {
        ...form,
        emission: predictionResult,
        userId: currentUser.uid,
        createdAt: serverTimestamp(),
        userEmail: currentUser.email,
      });

      setSaveStatus("Results saved successfully!");
    } catch (err) {
      console.error("Error:", err);
      if (err.response) {
        // API error
        setError(`Prediction error: ${err.response.data.message || err.message}`);
      } else if (err.code && err.code.startsWith("firestore")) {
        // Firestore error
        setError(`Database error: ${err.message}`);
      } else {
        // General error
        setError(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-container">
      <h2>Carbon Emission Calculator</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="number"
          name="distance"
          placeholder="Monthly Distance (km)"
          value={form.distance}
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="transport_type"
          placeholder="Transport Type (e.g. 0 for car)"
          value={form.transport_type}
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="body_type"
          placeholder="Body Type (e.g. 1 for SUV)"
          value={form.body_type}
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="sex"
          placeholder="Sex (e.g. 0 for male)"
          value={form.sex}
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="diet"
          placeholder="Diet (e.g. 1 for vegetarian)"
          value={form.diet}
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="grocery"
          placeholder="Monthly Grocery Bill (GHS)"
          value={form.grocery}
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="screen_time"
          placeholder="Daily TV/PC Time (hours)"
          value={form.screen_time}
          onChange={handleChange}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Calculating..." : "Estimate Emissions"}
        </button>
      </form>

      {emission && (
        <div className="result">
          <p>
            Estimated CO₂: <strong>{emission} kg</strong>
          </p>
          {saveStatus && <p className="success">{saveStatus}</p>}
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default CarbonForm;
