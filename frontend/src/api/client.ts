import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/v1`
  : '/api/v1'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截：自动注入 Bearer token
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('etf_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截：401 时清除登录状态并跳转登录页
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('etf_token')
      localStorage.removeItem('etf_user')
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/login'
      }
    } else if (error.response) {
      const { status, data } = error.response
      console.error(`API Error [${status}]:`, data?.detail || data?.error?.message || data)
    } else if (error.request) {
      console.error('Network Error: No response received')
    } else {
      console.error('Request Error:', error.message)
    }
    return Promise.reject(error)
  }
)

export default client

// API response type
export interface ApiResponse<T> {
  success: boolean
  data: T
  error: null | { code: string; message: string }
}

// Backtest types
export interface BacktestParams {
  strategy_type: 'classic' | 'hunter' | 'steady'
  start_date: string
  end_date: string
  initial_capital: number
  top_n: number
  weight_method: 'equal' | 'momentum_weighted' | 'inverse_volatility'
  rebalance_freq: 'weekly' | 'biweekly' | 'monthly'
  momentum_window: number
  stop_loss_enabled: boolean
  stop_loss_threshold: number
  trailing_stop: boolean
  trailing_stop_threshold: number
  selected_codes: string[] | null
}

export interface StrategyInfo {
  id: string
  name: string
  description: string
  configurable: boolean
}

export interface BacktestResult {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  metrics: BacktestMetrics | null
  nav_history: NavPoint[]
  benchmark_history: NavPoint[]
  trades: TradeRecord[]
}

export interface BacktestMetrics {
  total_return: number
  annual_return: number
  max_drawdown: number
  sharpe_ratio: number
  calmar_ratio: number
  win_rate: number
  total_trades: number
  profit_factor: number
  volatility: number
  benchmark_return: number
  alpha: number
  beta: number
}

export interface NavPoint {
  date: string
  value: number
}

export interface TradeRecord {
  date: string
  etf_code: string
  etf_name: string
  direction: 'buy' | 'sell'
  price: number
  volume: number
  amount: number
  reason: string
}

// Signal types
export interface Signal {
  id: string
  date: string
  market_regime: 'bull' | 'bear' | 'sideways'
  cash_ratio: number
  recommended_etfs: RecommendedETF[]
  status: 'pending' | 'adopted' | 'ignored'
  created_at: string
}

export interface RecommendedETF {
  code: string
  name: string
  industry: string
  score: number
  momentum_score: number
  volume_score: number
  weight: number
}

// Portfolio types
export interface Position {
  etf_code: string
  etf_name: string
  industry: string
  shares: number
  avg_cost: number
  current_price: number
  market_value: number
  pnl: number
  pnl_pct: number
  weight: number
}

export interface PortfolioPerformance {
  nav_history: NavPoint[]
  total_value: number
  total_pnl: number
  total_pnl_pct: number
  cash: number
  positions_value: number
}

export interface PortfolioTrade {
  id: string
  date: string
  etf_code: string
  etf_name: string
  direction: 'buy' | 'sell'
  price: number
  volume: number
  amount: number
  signal_id: string | null
}
