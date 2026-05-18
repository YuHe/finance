import type { BacktestMetrics } from '../api/client'

interface MetricsPanelProps {
  metrics: BacktestMetrics | null
  loading?: boolean
}

interface MetricItem {
  label: string
  key: keyof BacktestMetrics
  format: (v: number) => string
  colorFn?: (v: number) => string
}

const metricItems: MetricItem[] = [
  {
    label: '总收益率',
    key: 'total_return',
    format: (v) => `${(v * 100).toFixed(2)}%`,
    colorFn: (v) => (v >= 0 ? 'text-red-400' : 'text-green-400'),
  },
  {
    label: '年化收益',
    key: 'annual_return',
    format: (v) => `${(v * 100).toFixed(2)}%`,
    colorFn: (v) => (v >= 0 ? 'text-red-400' : 'text-green-400'),
  },
  {
    label: '最大回撤',
    key: 'max_drawdown',
    format: (v) => `${(v * 100).toFixed(2)}%`,
    colorFn: () => 'text-yellow-400',
  },
  {
    label: '夏普比率',
    key: 'sharpe_ratio',
    format: (v) => v.toFixed(3),
    colorFn: (v) => (v >= 1 ? 'text-green-400' : v >= 0 ? 'text-yellow-400' : 'text-red-400'),
  },
  {
    label: '卡玛比率',
    key: 'calmar_ratio',
    format: (v) => v.toFixed(3),
    colorFn: (v) => (v >= 1 ? 'text-green-400' : 'text-yellow-400'),
  },
  {
    label: '胜率',
    key: 'win_rate',
    format: (v) => `${(v * 100).toFixed(1)}%`,
    colorFn: (v) => (v >= 0.5 ? 'text-green-400' : 'text-red-400'),
  },
  {
    label: '总交易次数',
    key: 'total_trades',
    format: (v) => v.toString(),
  },
  {
    label: '盈亏比',
    key: 'profit_factor',
    format: (v) => v.toFixed(2),
    colorFn: (v) => (v >= 1.5 ? 'text-green-400' : v >= 1 ? 'text-yellow-400' : 'text-red-400'),
  },
  {
    label: '波动率',
    key: 'volatility',
    format: (v) => `${(v * 100).toFixed(2)}%`,
  },
  {
    label: '基准收益',
    key: 'benchmark_return',
    format: (v) => `${(v * 100).toFixed(2)}%`,
    colorFn: (v) => (v >= 0 ? 'text-red-400' : 'text-green-400'),
  },
  {
    label: 'Alpha',
    key: 'alpha',
    format: (v) => `${(v * 100).toFixed(2)}%`,
    colorFn: (v) => (v >= 0 ? 'text-green-400' : 'text-red-400'),
  },
  {
    label: 'Beta',
    key: 'beta',
    format: (v) => v.toFixed(3),
  },
]

function MetricsPanel({ metrics, loading }: MetricsPanelProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-4 animate-pulse">
            <div className="h-3 bg-gray-700 rounded w-16 mb-2" />
            <div className="h-6 bg-gray-700 rounded w-20" />
          </div>
        ))}
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-500 text-sm">
        运行回测后查看绩效指标
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
      {metricItems.map((item) => {
        const value = metrics[item.key]
        const colorClass = item.colorFn ? item.colorFn(value) : 'text-white'
        return (
          <div
            key={item.key}
            className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors"
          >
            <div className="text-xs text-gray-400 mb-1">{item.label}</div>
            <div className={`text-xl font-bold ${colorClass}`}>
              {item.format(value)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default MetricsPanel
