import React, { useState } from 'react';
import './EcoTips.css';

const EcoTips = ({ tips }) => {
  const [expandedTip, setExpandedTip] = useState(null);

  const toggleTip = (index) => {
    setExpandedTip(expandedTip === index ? null : index);
  };

  if (!tips || tips.length === 0) return null;

  return (
    <div className="eco-tips">
      <h3>
        <span className="eco-icon">🌍</span>
        Eco-Friendly Suggestions
      </h3>
      <div className="tips-grid">
        {tips.map((tip, index) => (
          <div 
            key={index} 
            className={`tip-card ${expandedTip === index ? 'expanded' : ''}`}
            onClick={() => toggleTip(index)}
          >
            <div className="tip-header">
              <div className="category-icon" style={{ backgroundColor: tip.category.color }}>
                {tip.category.icon}
              </div>
              <h4>{tip.category.name}</h4>
              <span className={`impact-badge ${tip.impact.toLowerCase().replace(' ', '-')}`}>
                {tip.impact}
              </span>
            </div>
            <p className="tip-content">{tip.tip}</p>
            {tip.actions && (
              <div className={`tip-actions ${expandedTip === index ? 'show' : ''}`}>
                <h5>Suggested Actions:</h5>
                <ul>
                  {tip.actions.map((action, actionIndex) => (
                    <li key={actionIndex}>
                      <span className="action-bullet">•</span>
                      {action}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div className="expand-indicator">
              {expandedTip === index ? '↑ Less' : '↓ More'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EcoTips;
