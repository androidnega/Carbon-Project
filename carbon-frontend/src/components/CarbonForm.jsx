import React, { useState } from "react";
import axios from "axios";
import "./CarbonForm.css";

const CarbonForm = () => {
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

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await axios.post("http://localhost:5000/api/predict", form);
      setEmission(res.data.emission);
    } catch (err) {
      setError("Error getting prediction. Check your input or server.");
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
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="transport_type"
          placeholder="Transport Type (e.g. 0 for car)"
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="body_type"
          placeholder="Body Type (e.g. 1 for SUV)"
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="sex"
          placeholder="Sex (e.g. 0 for male)"
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="diet"
          placeholder="Diet (e.g. 1 for vegetarian)"
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="grocery"
          placeholder="Monthly Grocery Bill (GHS)"
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="screen_time"
          placeholder="Daily TV/PC Time (hours)"
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
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default CarbonForm;
