import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyA0WuMaIZEPqoZeUPoaCY5Z18sLGK1n34c",
  authDomain: "carbon-footprint-calcula-2823f.firebaseapp.com",
  projectId: "carbon-footprint-calcula-2823f",
  storageBucket: "carbon-footprint-calcula-2823f.firebasestorage.app",
  messagingSenderId: "228102789007",
  appId: "1:228102789007:web:75dfce1ecf780cf642ddfa"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and Firestore
const auth = getAuth(app);
const db = getFirestore(app);

export { auth, db };
export default app;
