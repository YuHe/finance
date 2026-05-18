import { useState, useCallback } from 'react'
import client from '../api/client'
import type { ApiResponse, BacktestParams, BacktestResult, TradeRecord } from '../api/client'
import BacktestConfig from '../components/BacktestConfig'
import NavChart from '../components/NavChart'
import MetricsPanel from '../components/MetricsPanel'
import dayjs from 'dayjs'

function BacktestPage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const pollResult = useCallback(async (id: string) => {
    const maxAttempts = 60
    let attempts = 0

    while (attempts < maxAttempts) {
      try {
        const response = await client.get<ApiResponse<BacktestResult>>(`/backtest/result/${id}`)
        const data = response.data.data

        if (data.status === 'completed') {
          setResult(data)
          setLoading(false)
          return
        } else if (data.status === 'failed') {
          setError('回测运行失败，请检查参数后重试')
          setLoading(false)
          return
        }

        // Still running, wait and poll again
        await new Promise((resolve) => setTimeout(resolve, 2000))
        attempts++
      } catch (err) {
        console.error('Poll error:', err)
        setError('获取回测结果失败')
        setLoading(false)
        return
      }
    }

    setError('回测超时，请稍后重试')
    setLoading(false)
  }, [])

  const handleSubmit = async (params: BacktestParams) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await client.post<ApiResponse<{ id: string }>>('/backtest/run', params)
      if (response.data.success) {
        await pollResult(response.data.data.id)
      } else {
        setError(response.data.error?.message || '提交回测请求失败')
        setLoading(false)
      }
    } catch (err) {
      console.error('Submit error:', err)
      setError('提交回测请求失败，请检查后端服务是否运行')
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full">
      {/* Left Panel - Config */}
      <div className="w-80 flex-shrink-0 border-r border-gray-700 overflow-y-auto p-5">
        <BacktestConfig onSubmit={handleSubmit} loading={loading} />
      </div>

      {/* Right Panel - Results */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-white">回测结果</h1>
          {result && (
            <span className="text-xs text-gray-400">
              回测ID: {result.id}
            </span>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Nav Chart */}
        <NavChart
          strategyNav={result?.nav_history || []}
          benchmarkNav={result?.benchmark_history || []}
          loading={loading}
        />

        {/* Metrics */}
        <MetricsPanel metrics={result?.metrics || null} loading={loading} />

        {/* Trade List */}
        {result?.trades && result.trades.length > 0 && (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="p-4 border-b border-gray-700">
              <h3 className="text-sm font-semibold text-white">交易记录</h3>
              <p className="text-xs text-gray-400 mt-1">共 {result.trades.length} 笔交易</p>
            </div>
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-800">
                  <tr className="text-xs text-gray-400 border-b border-gray-700">
                    <th className="text-left px-4 py-3 font-medium">日期</th>
                    <th className="text-left px-4 py-3 font-medium">ETF</th>
                    <th className="text-center px-4 py-3 font-medium">方向</th>
                    <th className="text-right px-4 py-3 font-medium">价格</th>
                    <th className="text-right px-4 py-3 font-medium">数量</th>
                    <th className="text-right px-4 py-3 font-medium">金额</th>
                    <th className="text-left px-4 py-3 font-medium">原因</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.map((trade: TradeRecord, idx: number) => (
                    <tr
                      key={idx}
                      className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors"
                    >
                      <td className="px-4 py-2.5 text-gray-300">
                        {dayjs(trade.date).format('YYYY-MM-DD')}
                      </td>
                      <td className="px-4 py-2.5">
                        <span className="text-white">{trade.etf_name}</span>
                        <span className="text-xs text-gray-500 ml-1">{trade.etf_code}</span>
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        <span
                          className={`text-xs px-2 py-0.5 rounded ${
                            trade.direction === 'buy'
                              ? 'bg-red-500/20 text-red-400'
                              : 'bg-green-500/20 text-green-400'
                          }`}
                        >
                          {trade.direction === 'buy' ? '买入' : '卖出'}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-right text-gray-300">
                        {trade.price.toFixed(3)}
                      </td>
                      <td className="px-4 py-2.5 text-right text-gray-300">
                        {trade.volume.toLocaleString()}
                      </td>
                      <td className="px-4 py-2.5 text-right text-gray-300">
                        {trade.amount.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
                      </td>
                      <td className="px-4 py-2.5 text-gray-400 text-xs">
                        {trade.reason}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!loading && !result && !error && (
          <div className="flex flex-col items-center justify-center py-20 text-gray-500">
            <svg className="w-16 h-16 mb-4 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="text-sm">配置参数后点击"开始回测"运行策略回测</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default BacktestPage
