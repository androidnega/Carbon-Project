import React from 'react';
import { PayPalButtons } from "@paypal/react-paypal-js";
import "./CarbonOffset.css";

const CarbonOffset = ({ emission, onSuccess }) => {
  // Calculate offset amount (1 USD per 100kg CO₂)
  const calculateOffsetAmount = () => {
    return Math.max(1, Math.ceil(emission / 100));
  };

  return (
    <div className="offset-container">
      <h4>Offset Your Carbon Footprint</h4>
      <div className="offset-info">
        <div className="offset-details">
          <p>Your carbon footprint: <strong>{emission} kg CO₂</strong></p>
          <p>Suggested offset amount: <strong>${calculateOffsetAmount()} USD</strong></p>
          <p className="offset-explanation">
            💚 Your contribution goes towards environmental projects that help reduce carbon emissions.
          </p>
        </div>
        <div className="offset-benefits">
          <h5>Benefits of Offsetting</h5>
          <ul>
            <li>🌳 Support reforestation projects</li>
            <li>🌞 Fund renewable energy initiatives</li>
            <li>♻️ Promote sustainable development</li>
            <li>🌍 Combat climate change</li>
          </ul>
        </div>
      </div>
      
      <div className="paypal-container">
        <PayPalButtons
          style={{
            layout: "vertical",
            color: "blue",
            shape: "rect",
            label: "pay"
          }}
          createOrder={(data, actions) => {
            const offsetAmount = calculateOffsetAmount();
            return actions.order.create({
              purchase_units: [
                {
                  amount: {
                    value: offsetAmount.toString(),
                    currency_code: "USD",
                  },
                  description: `Carbon Offset for ${emission} kg CO₂`,
                },
              ],
            });
          }}
          onApprove={(data, actions) => {
            return actions.order.capture().then((details) => {
              onSuccess(details);
            });
          }}
        />
      </div>
    </div>
  );
};

export default CarbonOffset;
