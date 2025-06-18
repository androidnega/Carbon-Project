import { AuthProvider } from './firebase/AuthContext'
import { useAuth } from './firebase/AuthContext'
import Login from './components/Login'
import CarbonForm from './components/CarbonForm'
import './App.css'

function AppContent() {
  const { currentUser } = useAuth();

  return (
    <div className="App">
      {currentUser ? <CarbonForm /> : <Login />}
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
