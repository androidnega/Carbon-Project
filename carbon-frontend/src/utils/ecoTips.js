// Thresholds for different categories
export const THRESHOLDS = {
  DISTANCE: {
    HIGH: 2000,
    MEDIUM: 1000,
  },
  SCREEN_TIME: {
    HIGH: 6,
    MEDIUM: 4,
  },
  GROCERY: {
    HIGH: 800,
    MEDIUM: 500,
  }
};

// Tip categories with their respective icons and colors
export const CATEGORIES = {
  TRANSPORT: {
    name: "Transportation",
    icon: "🚗",
    color: "#4CAF50"
  },
  DIET: {
    name: "Diet & Food",
    icon: "🥗",
    color: "#8BC34A"
  },
  ENERGY: {
    name: "Energy Usage",
    icon: "⚡",
    color: "#FFC107"
  },
  LIFESTYLE: {
    name: "Lifestyle",
    icon: "🌱",
    color: "#009688"
  },
  SHOPPING: {
    name: "Shopping",
    icon: "🛍️",
    color: "#FF5722"
  }
};

// Generate tips based on user input
export const generateEcoTips = (userData) => {
  const tips = [];

  // Transportation tips
  const distance = parseFloat(userData.distance);
  if (distance > THRESHOLDS.DISTANCE.HIGH) {
    tips.push({
      category: CATEGORIES.TRANSPORT,
      tip: "Consider switching to public transportation or carpooling to significantly reduce your carbon footprint.",
      impact: "High Impact",
      actions: [
        "Research local public transport routes",
        "Join local carpooling groups",
        "Consider remote work options if possible"
      ]
    });
  } else if (distance > THRESHOLDS.DISTANCE.MEDIUM) {
    tips.push({
      category: CATEGORIES.TRANSPORT,
      tip: "Try combining trips or using a bike for shorter distances.",
      impact: "Medium Impact",
      actions: [
        "Plan weekly trips in advance",
        "Invest in a bicycle for short trips",
        "Walk for distances under 2km"
      ]
    });
  }

  // Diet-related tips
  if (userData.diet === "0") { // Non-vegetarian
    tips.push({
      category: CATEGORIES.DIET,
      tip: "Gradually incorporate more plant-based meals into your diet.",
      impact: "High Impact",
      actions: [
        "Try Meatless Monday",
        "Explore vegetarian recipes",
        "Choose local and seasonal produce"
      ]
    });
  }

  // Grocery shopping tips
  const grocery = parseFloat(userData.grocery);
  if (grocery > THRESHOLDS.GROCERY.HIGH) {
    tips.push({
      category: CATEGORIES.SHOPPING,
      tip: "Optimize your grocery shopping to reduce waste and emissions.",
      impact: "Medium Impact",
      actions: [
        "Make a shopping list and stick to it",
        "Buy in bulk to reduce packaging",
        "Choose products with minimal packaging"
      ]
    });
  }

  // Screen time and energy usage tips
  const screenTime = parseFloat(userData.screen_time);
  if (screenTime > THRESHOLDS.SCREEN_TIME.HIGH) {
    tips.push({
      category: CATEGORIES.ENERGY,
      tip: "Reduce your digital carbon footprint by optimizing screen time and device usage.",
      impact: "Medium Impact",
      actions: [
        "Use energy-saving display settings",
        "Unplug devices when not in use",
        "Take regular screen breaks"
      ]
    });
  }

  // Lifestyle tips (based on combination of factors)
  if (screenTime > THRESHOLDS.SCREEN_TIME.MEDIUM && distance > THRESHOLDS.DISTANCE.MEDIUM) {
    tips.push({
      category: CATEGORIES.LIFESTYLE,
      tip: "Consider adopting a more active and eco-conscious lifestyle.",
      impact: "High Impact",
      actions: [
        "Replace some screen time with outdoor activities",
        "Join local environmental groups",
        "Start a home garden"
      ]
    });
  }

  return tips;
};

// Get color based on impact level
export const getImpactColor = (impact) => {
  switch (impact.toLowerCase()) {
    case "high impact":
      return "#2e7d32";
    case "medium impact":
      return "#ef6c00";
    case "low impact":
      return "#757575";
    default:
      return "#757575";
  }
};
