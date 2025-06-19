import { AuthProvider } from './firebase/AuthContext'
import { useAuth } from './firebase/AuthContext'
import { PayPalScriptProvider } from "@paypal/react-paypal-js";
import Login from './components/Login'
import CarbonForm from './components/CarbonForm'
import EmissionHistory from './components/EmissionHistory'
import './App.css'

const PAYPAL_CLIENT_ID = import.meta.env.VITE_PAYPAL_CLIENT_ID;

function AppContent() {
  const { currentUser } = useAuth();

  return (
    <div className="App">
      {!currentUser ? (
        <Login />
      ) : (
        <>
          <CarbonForm />
          <EmissionHistory />
        </>
      )}
    </div>
  );
}

function App() {
  return (
    <PayPalScriptProvider options={{ 
      "client-id": PAYPAL_CLIENT_ID,
      currency: "USD"
    }}>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </PayPalScriptProvider>
  )
}

export default App
