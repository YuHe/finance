import { useState, useEffect, useCallback } from 'react'
import client from '../api/client'
import type { ApiResponse, Signal } from '../api/client'
import SignalCard from '../components/SignalCard'
import dayjs from 'dayjs'

function SignalPage() {
  const [latestSignal, setLatestSignal] = useState<Signal | null>(null)
  const [signalHistory, setSignalHistory] = useState<Signal[]>([])
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchLatestSignal = useCallback(async () => {
    try {
      const response = await client.get<ApiResponse<Signal>>('/signal/latest')
      if (response.data.success) {
        setLatestSignal(response.data.data)
      }
    } catch (err) {
      console.error('Fetch latest signal error:', err)
      setError('获取最新信号失败')
    }
  }, [])

  const fetchSignalHistory = useCallback(async () => {
    try {
      const response = await client.get<ApiResponse<Signal[]>>('/signal/history')
      if (response.data.success) {
        setSignalHistory(response.data.data)
      }
    } catch (err) {
      console.error('Fetch signal history error:', err)
    }
  }, [])

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchLatestSignal(), fetchSignalHistory()])
      setLoading(false)
    }
    loadData()
  }, [fetchLatestSignal, fetchSignalHistory])

  const handleAdopt = async (id: string) => {
    setActionLoading(true)
    try {
      const response = await client.post<ApiResponse<null>>(`/signal/${id}/adopt`)
      if (response.data.success) {
        setLatestSignal((prev) => (prev ? { ...prev, status: 'adopted' } : prev))
        setSignalHistory((prev) =>
          prev.map((s) => (s.id === id ? { ...s, status: 'adopted' as const } : s))
        )
      }
    } catch (err) {
      console.error('Adopt signal error:', err)
      setError('采纳信号失败')
    } finally {
      setActionLoading(false)
    }
  }

  const handleIgnore = async (id: string) => {
    setActionLoading(true)
    try {
      const response = await client.post<ApiResponse<null>>(`/signal/${id}/ignore`)
      if (response.data.success) {
        setLatestSignal((prev) => (prev ? { ...prev, status: 'ignored' } : prev))
        setSignalHistory((prev) =>
          prev.map((s) => (s.id === id ? { ...s, status: 'ignored' as const } : s))
        )
      }
    } catch (err) {
      console.error('Ignore signal error:', err)
      setError('忽略信号失败')
    } finally {
      setActionLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-800 rounded w-48" />
          <div className="h-64 bg-gray-800 rounded-lg" />
          <div className="h-48 bg-gray-800 rounded-lg" />
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <h1 className="text-xl font-bold text-white">交易信号</h1>

      {/* Error */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-3 text-red-300 hover:text-red-200 underline text-xs"
          >
            关闭
          </button>
        </div>
      )}

      {/* Latest Signal */}
      {latestSignal ? (
        <SignalCard
          signal={latestSignal}
          onAdopt={handleAdopt}
          onIgnore={handleIgnore}
          loading={actionLoading}
        />
      ) : (
        <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-500 text-sm">
          暂无信号数据
        </div>
      )}

      {/* Signal History */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-sm font-semibold text-white">历史信号</h3>
        </div>
        {signalHistory.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-gray-400 border-b border-gray-700">
                  <th className="text-left px-4 py-3 font-medium">日期</th>
                  <th className="text-center px-4 py-3 font-medium">市场状态</th>
                  <th className="text-center px-4 py-3 font-medium">现金比例</th>
                  <th className="text-left px-4 py-3 font-medium">推荐ETF</th>
                  <th className="text-center px-4 py-3 font-medium">状态</th>
                </tr>
              </thead>
              <tbody>
                {signalHistory.map((signal) => (
                  <tr
                    key={signal.id}
                    className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors"
                  >
                    <td className="px-4 py-3 text-gray-300">
                      {dayjs(signal.date).format('YYYY-MM-DD')}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          signal.market_regime === 'bull'
                            ? 'bg-red-500/20 text-red-400'
                            : signal.market_regime === 'bear'
                            ? 'bg-green-500/20 text-green-400'
                            : 'bg-yellow-500/20 text-yellow-400'
                        }`}
                      >
                        {signal.market_regime === 'bull'
                          ? '牛市'
                          : signal.market_regime === 'bear'
                          ? '熊市'
                          : '震荡'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center text-blue-400">
                      {(signal.cash_ratio * 100).toFixed(0)}%
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {signal.recommended_etfs.slice(0, 3).map((etf) => (
                          <span
                            key={etf.code}
                            className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded"
                          >
                            {etf.name}
                          </span>
                        ))}
                        {signal.recommended_etfs.length > 3 && (
                          <span className="text-xs text-gray-500">
                            +{signal.recommended_etfs.length - 3}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          signal.status === 'adopted'
                            ? 'bg-blue-500/20 text-blue-400'
                            : signal.status === 'ignored'
                            ? 'bg-gray-600/50 text-gray-400'
                            : 'bg-yellow-500/20 text-yellow-400'
                        }`}
                      >
                        {signal.status === 'adopted'
                          ? '已采纳'
                          : signal.status === 'ignored'
                          ? '已忽略'
                          : '待处理'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500 text-sm">暂无历史信号</div>
        )}
      </div>
    </div>
  )
}

export default SignalPage
