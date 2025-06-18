import { useState } from 'react';
import { useAuth } from '../firebase/AuthContext';
import './Login.css';

export default function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login, signInWithGoogle } = useAuth();

    async function handleSubmit(e) {
        e.preventDefault();

        try {
            setError('');
            setLoading(true);
            await login(email, password);
        } catch (err) {
            setError('Failed to sign in: ' + err.message);
        } finally {
            setLoading(false);
        }
    }

    async function handleGoogleSignIn() {
        try {
            setError('');
            setLoading(true);
            await signInWithGoogle();
        } catch (err) {
            setError('Failed to sign in with Google: ' + err.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="login-container">
            <div className="login-box">
                <h2>Login</h2>
                {error && <div className="alert error">{error}</div>}
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Email</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    <button disabled={loading} type="submit" className="login-button">
                        {loading ? 'Logging in...' : 'Login'}
                    </button>
                    <button 
                        type="button" 
                        onClick={handleGoogleSignIn}
                        disabled={loading}
                        className="google-button"
                    >
                        Sign in with Google
                    </button>
                </form>
            </div>
        </div>
    );
}
