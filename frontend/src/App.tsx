import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './store/authStore'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import BacktestPage from './pages/BacktestPage'
import SignalPage from './pages/SignalPage'
import PortfolioPage from './pages/PortfolioPage'
import DataPage from './pages/DataPage'
import AdminPage from './pages/AdminPage'
import LLMSettingsPage from './pages/LLMSettingsPage'

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { token } = useAuth()
  if (!token) return <Navigate to="/login" replace />
  return <>{children}</>
}

function RequireAdmin({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  if (!user?.is_admin) return <Navigate to="/" replace />
  return <>{children}</>
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/backtest" replace />} />
        <Route path="backtest" element={<BacktestPage />} />
        <Route path="data" element={<DataPage />} />
        <Route path="signal" element={<SignalPage />} />
        <Route path="portfolio" element={<PortfolioPage />} />
        <Route path="llm" element={<LLMSettingsPage />} />
        <Route
          path="admin"
          element={
            <RequireAdmin>
              <AdminPage />
            </RequireAdmin>
          }
        />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App
