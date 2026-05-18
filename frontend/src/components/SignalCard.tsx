import type { Signal } from '../api/client'
import dayjs from 'dayjs'

interface SignalCardProps {
  signal: Signal
  onAdopt: (id: string) => void
  onIgnore: (id: string) => void
  loading?: boolean
}

const regimeLabels: Record<string, { text: string; color: string }> = {
  bull: { text: '牛市', color: 'bg-red-500/20 text-red-400 border-red-500/30' },
  bear: { text: '熊市', color: 'bg-green-500/20 text-green-400 border-green-500/30' },
  sideways: { text: '震荡', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
}

function SignalCard({ signal, onAdopt, onIgnore, loading }: SignalCardProps) {
  const regime = regimeLabels[signal.market_regime] || regimeLabels.sideways

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold text-white">本周信号</h3>
          <span className={`text-xs px-2 py-1 rounded border ${regime.color}`}>
            {regime.text}
          </span>
        </div>
        <div className="text-xs text-gray-400">
          {dayjs(signal.created_at).format('YYYY-MM-DD HH:mm')}
        </div>
      </div>

      {/* Cash Ratio */}
      <div className="mb-4 p-3 bg-gray-900 rounded-lg">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">建议现金比例</span>
          <span className="text-lg font-bold text-blue-400">
            {(signal.cash_ratio * 100).toFixed(0)}%
          </span>
        </div>
        <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all"
            style={{ width: `${signal.cash_ratio * 100}%` }}
          />
        </div>
      </div>

      {/* Recommended ETFs */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-300 mb-2">推荐ETF</h4>
        <div className="space-y-2">
          {signal.recommended_etfs.map((etf) => (
            <div
              key={etf.code}
              className="flex items-center justify-between p-3 bg-gray-900 rounded-lg"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-white">{etf.name}</span>
                  <span className="text-xs text-gray-500">{etf.code}</span>
                </div>
                <div className="text-xs text-gray-400 mt-1">{etf.industry}</div>
              </div>
              <div className="text-right">
                <div className="text-sm font-bold text-blue-400">
                  {etf.score.toFixed(2)}
                </div>
                <div className="text-xs text-gray-500">
                  权重 {(etf.weight * 100).toFixed(1)}%
                </div>
              </div>
              <div className="ml-4 flex flex-col gap-1">
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-500">动量</span>
                  <div className="w-16 h-1.5 bg-gray-700 rounded-full">
                    <div
                      className="h-full bg-blue-500 rounded-full"
                      style={{ width: `${Math.min(etf.momentum_score * 100, 100)}%` }}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-500">量能</span>
                  <div className="w-16 h-1.5 bg-gray-700 rounded-full">
                    <div
                      className="h-full bg-purple-500 rounded-full"
                      style={{ width: `${Math.min(etf.volume_score * 100, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      {signal.status === 'pending' && (
        <div className="flex gap-3">
          <button
            onClick={() => onAdopt(signal.id)}
            disabled={loading}
            className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-medium rounded-lg transition-colors text-sm"
          >
            采纳
          </button>
          <button
            onClick={() => onIgnore(signal.id)}
            disabled={loading}
            className="flex-1 py-2.5 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-600 text-gray-300 font-medium rounded-lg transition-colors text-sm"
          >
            忽略
          </button>
        </div>
      )}
      {signal.status === 'adopted' && (
        <div className="text-center py-2 text-sm text-green-400 bg-green-500/10 rounded-lg">
          已采纳
        </div>
      )}
      {signal.status === 'ignored' && (
        <div className="text-center py-2 text-sm text-gray-400 bg-gray-700/50 rounded-lg">
          已忽略
        </div>
      )}
    </div>
  )
}

export default SignalCard
