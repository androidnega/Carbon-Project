import { createContext, useContext, useState, useEffect } from 'react';
import { 
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut,
    onAuthStateChanged,
    GoogleAuthProvider,
    signInWithPopup,
    sendPasswordResetEmail
} from 'firebase/auth';
import { auth } from './config';

const AuthContext = createContext();

export function useAuth() {
    return useContext(AuthContext);
}

export function AuthProvider({ children }) {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // Sign Up with email and password
    function signup(email, password) {
        return createUserWithEmailAndPassword(auth, email, password);
    }

    // Login with email and password
    function login(email, password) {
        return signInWithEmailAndPassword(auth, email, password);
    }

    // Logout
    function logout() {
        return signOut(auth);
    }

    // Reset Password
    function resetPassword(email) {
        return sendPasswordResetEmail(auth, email);
    }

    // Google Sign In
    function signInWithGoogle() {
        const provider = new GoogleAuthProvider();
        return signInWithPopup(auth, provider);
    }

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            setCurrentUser(user);
            setLoading(false);
        });

        return unsubscribe;
    }, []);

    const value = {
        currentUser,
        signup,
        login,
        logout,
        resetPassword,
        signInWithGoogle
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
}
