import { createContext, useContext, useState, ReactNode } from 'react'
import type { UserInfo } from '../api/auth'

const TOKEN_KEY = 'etf_token'
const USER_KEY = 'etf_user'

interface AuthState {
  token: string | null
  user: UserInfo | null
  login: (token: string, user: UserInfo) => void
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY))
  const [user, setUser] = useState<UserInfo | null>(() => {
    const s = localStorage.getItem(USER_KEY)
    return s ? JSON.parse(s) : null
  })

  const login = (t: string, u: UserInfo) => {
    localStorage.setItem(TOKEN_KEY, t)
    localStorage.setItem(USER_KEY, JSON.stringify(u))
    setToken(t)
    setUser(u)
  }

  const logout = () => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ token, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
