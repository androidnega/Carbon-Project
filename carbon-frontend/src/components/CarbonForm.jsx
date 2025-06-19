import React, { useState } from "react";
import axios from "axios";
import { db } from "../firebase/config";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { useAuth } from "../firebase/AuthContext";
import EcoTips from "./EcoTips";
import CarbonOffset from "./CarbonOffset";
import SustainabilityReport from "./SustainabilityReport";
import { generateEcoTips } from "../utils/ecoTips";
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
  const [ecoTips, setEcoTips] = useState([]);
  const [offsetComplete, setOffsetComplete] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleOffsetSuccess = async (details) => {
    try {
      // Save offset payment details to Firestore
      await addDoc(collection(db, "offset_payments"), {
        userId: currentUser.uid,
        userEmail: currentUser.email,
        emission: emission,
        paymentAmount: details.purchase_units[0].amount.value,
        paymentId: details.id,
        payerName: details.payer.name,
        timestamp: serverTimestamp(),
      });

      setOffsetComplete(true);
      setSaveStatus("Carbon offset payment completed successfully!");
    } catch (err) {
      setError("Failed to record offset payment.");
      console.error("Offset payment record error:", err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSaveStatus(null);
    setEcoTips([]);
    setOffsetComplete(false);

    try {
      // Get prediction from API
      const res = await axios.post("http://localhost:5000/api/predict", form);
      const predictionResult = res.data.emission;
      setEmission(predictionResult);

      // Generate eco-friendly tips
      const tips = generateEcoTips(form);
      setEcoTips(tips);

      // Save to Firestore
      await addDoc(collection(db, "emission_records"), {
        ...form,
        emission: predictionResult,
        userId: currentUser.uid,
        createdAt: serverTimestamp(),
        userEmail: currentUser.email,
        tips: tips,
      });

      setSaveStatus("Results saved successfully!");
    } catch (err) {
      console.error("Error:", err);
      if (err.response) {
        setError(`Prediction error: ${err.response.data.message || err.message}`);
      } else if (err.code && err.code.startsWith("firestore")) {
        setError(`Database error: ${err.message}`);
      } else {
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

      <EcoTips tips={ecoTips} />

      {emission && !offsetComplete && (
        <CarbonOffset
          emission={emission}
          onSuccess={handleOffsetSuccess}
        />
      )}

      {offsetComplete && (
        <div className="offset-success">
          <h4>🎉 Thank You for Offsetting!</h4>
          <p>
            Your contribution will help fund environmental projects to reduce
            carbon emissions.
          </p>
        </div>
      )}

      <SustainabilityReport />

      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default CarbonForm;
