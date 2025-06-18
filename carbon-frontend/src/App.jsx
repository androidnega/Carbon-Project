import { AuthProvider } from './firebase/AuthContext'
import Login from './components/Login'
import './App.css'

function App() {
  return (
    <AuthProvider>
      <Login />
    </AuthProvider>
  )
}

export default App
