import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../store/authStore'

const navItems = [
  { path: '/backtest', label: '回测', icon: '📊' },
  { path: '/signal', label: '信号', icon: '📡' },
  { path: '/portfolio', label: '模拟盘', icon: '💼' },
]

function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-lg font-bold text-blue-400">行业ETF轮动</h1>
          <p className="text-xs text-gray-400 mt-1">量化交易系统</p>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              <span className="text-lg">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
          {user?.is_admin && (
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-yellow-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`
              }
            >
              <span className="text-lg">⚙️</span>
              <span>用户管理</span>
            </NavLink>
          )}
        </nav>
        <div className="p-4 border-t border-gray-700 space-y-2">
          {user && (
            <div className="flex items-center gap-2 px-2">
              <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold flex-shrink-0">
                {user.username[0].toUpperCase()}
              </div>
              <div className="min-w-0">
                <p className="text-xs text-white font-medium truncate">{user.username}</p>
                <p className="text-xs text-gray-500">{user.is_admin ? '管理员' : '用户'}</p>
              </div>
            </div>
          )}
          <button
            onClick={handleLogout}
            className="w-full text-xs text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg px-3 py-2 text-left transition-colors"
          >
            退出登录
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout
