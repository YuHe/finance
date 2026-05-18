import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useState, useEffect, useRef } from 'react'
import client from '../api/client'
import { useAuth } from '../store/authStore'

const navItems = [
  { path: '/backtest', label: '回测', icon: '📊' },
  { path: '/data', label: '数据', icon: '💾' },
  { path: '/signal', label: '信号', icon: '📡' },
  { path: '/portfolio', label: '模拟盘', icon: '💼' },
  { path: '/llm', label: 'AI设置', icon: '🤖' },
]

type ItemStatus = 'pending' | 'running' | 'ok' | 'error'

interface UpdateItem {
  code: string
  name: string
  status: ItemStatus
  error: string
}

interface UpdateState {
  status: 'idle' | 'running' | 'done' | 'error'
  message: string
  progress: number
  total: number
  items: UpdateItem[]
}

const EMPTY_STATE: UpdateState = { status: 'idle', message: '', progress: 0, total: 0, items: [] }

function statusIcon(s: ItemStatus) {
  if (s === 'ok') return <span className="text-green-400">✓</span>
  if (s === 'error') return <span className="text-red-400">✕</span>
  if (s === 'running') return <span className="text-blue-400 animate-spin inline-block">↻</span>
  return <span className="text-gray-600">–</span>
}

function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [state, setState] = useState<UpdateState>(EMPTY_STATE)
  const [panelOpen, setPanelOpen] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  // 有错误时自动弹出的 toast
  const [errorToast, setErrorToast] = useState<string | null>(null)

  const stopPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  const startPoll = () => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await client.get<UpdateState>('/market/update/status')
        const d = res.data
        setState(d)
        if (d.status === 'done' || d.status === 'error') {
          stopPoll()
          // 如果有失败项，弹出 toast
          const errCount = d.items.filter(it => it.status === 'error').length
          if (errCount > 0) {
            setErrorToast(`${errCount} 个标的更新失败，请查看详情`)
          }
        }
      } catch {
        stopPoll()
        setState(prev => ({ ...prev, status: 'error', message: '网络错误' }))
      }
    }, 1000)
  }

  const handleUpdate = async () => {
    if (state.status === 'running') {
      setPanelOpen(true)
      return
    }
    setState({ ...EMPTY_STATE, status: 'running', message: '启动中...' })
    setPanelOpen(true)
    setErrorToast(null)
    try {
      await client.post('/market/update')
      startPoll()
    } catch {
      setState(prev => ({ ...prev, status: 'error', message: '启动失败' }))
    }
  }

  useEffect(() => () => stopPoll(), [])

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  const isRunning = state.status === 'running'
  const pct = state.total > 0 ? Math.round((state.progress / state.total) * 100) : 0
  const errItems = state.items.filter(it => it.status === 'error')

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
          {/* 数据更新按钮 */}
          <button
            onClick={handleUpdate}
            className={`w-full text-xs rounded-lg px-3 py-2 text-left transition-colors flex items-center gap-2 ${
              isRunning
                ? 'bg-blue-900/40 text-blue-300'
                : state.status === 'done'
                ? errItems.length > 0
                  ? 'bg-red-900/30 text-red-400 hover:bg-red-900/50'
                  : 'bg-green-900/30 text-green-400 hover:bg-green-900/50'
                : state.status === 'error'
                ? 'bg-red-900/40 text-red-400 hover:bg-red-900/60'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            <span className={isRunning ? 'animate-spin inline-block' : ''}>
              {state.status === 'done'
                ? errItems.length > 0 ? '⚠' : '✓'
                : state.status === 'error' ? '✕' : '↻'}
            </span>
            <span className="truncate flex-1">
              {isRunning
                ? state.total > 0
                  ? `${state.progress}/${state.total} 更新中`
                  : '启动中...'
                : state.status === 'done'
                ? errItems.length > 0 ? `完成(${errItems.length}个失败)` : '更新完成'
                : state.status === 'error'
                ? '更新失败'
                : '更新行情数据'}
            </span>
            {(isRunning || state.status === 'done') && (
              <span
                className="text-gray-400 hover:text-white"
                onClick={e => { e.stopPropagation(); setPanelOpen(v => !v) }}
              >
                {panelOpen ? '▲' : '▼'}
              </span>
            )}
          </button>

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
      <main className="flex-1 overflow-auto relative">
        <Outlet />
      </main>

      {/* 更新进度悬浮面板 */}
      {panelOpen && (isRunning || state.status === 'done' || state.status === 'error') && (
        <div className="fixed bottom-6 right-6 w-80 bg-gray-800 border border-gray-600 rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden">
          {/* 面板头部 */}
          <div className="flex items-center justify-between px-4 py-3 bg-gray-750 border-b border-gray-700">
            <span className="text-sm font-semibold text-white">行情数据更新</span>
            <button
              onClick={() => setPanelOpen(false)}
              className="text-gray-400 hover:text-white text-lg leading-none"
            >
              ×
            </button>
          </div>

          {/* 进度条 */}
          <div className="px-4 pt-3 pb-2">
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>{state.message}</span>
              <span>{state.total > 0 ? `${state.progress}/${state.total}` : ''}</span>
            </div>
            <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  state.status === 'done' && errItems.length > 0
                    ? 'bg-yellow-500'
                    : state.status === 'error'
                    ? 'bg-red-500'
                    : 'bg-blue-500'
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>

          {/* ETF 列表 */}
          {state.items.length > 0 && (
            <div className="overflow-y-auto max-h-72 px-3 pb-3 space-y-0.5">
              {state.items.map(item => (
                <div
                  key={item.code}
                  className={`flex items-start gap-2 px-2 py-1.5 rounded text-xs ${
                    item.status === 'error'
                      ? 'bg-red-900/30'
                      : item.status === 'ok'
                      ? 'bg-green-900/10'
                      : item.status === 'running'
                      ? 'bg-blue-900/20'
                      : ''
                  }`}
                >
                  <span className="w-4 text-center flex-shrink-0 mt-0.5">{statusIcon(item.status)}</span>
                  <div className="min-w-0 flex-1">
                    <span className={`font-medium ${
                      item.status === 'error' ? 'text-red-300'
                      : item.status === 'ok' ? 'text-green-300'
                      : item.status === 'running' ? 'text-blue-300'
                      : 'text-gray-500'
                    }`}>
                      {item.name}
                    </span>
                    <span className="text-gray-500 ml-1">{item.code}</span>
                    {item.status === 'error' && item.error && (
                      <p className="text-red-400 mt-0.5 break-all">{item.error}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 错误 Toast */}
      {errorToast && (
        <div
          className="fixed top-6 right-6 bg-red-900/90 border border-red-500 text-red-200 text-sm px-4 py-3 rounded-xl shadow-xl z-50 flex items-center gap-3 cursor-pointer"
          onClick={() => { setPanelOpen(true); setErrorToast(null) }}
        >
          <span>⚠</span>
          <span>{errorToast}</span>
          <button
            className="text-red-400 hover:text-white ml-2"
            onClick={e => { e.stopPropagation(); setErrorToast(null) }}
          >
            ×
          </button>
        </div>
      )}
    </div>
  )
}

export default Layout
