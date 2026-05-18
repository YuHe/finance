import { useState, useEffect, useCallback } from 'react'
import ReactECharts from 'echarts-for-react'
import client from '../api/client'
import type { ApiResponse, Position, PortfolioPerformance, PortfolioTrade } from '../api/client'
import PositionTable from '../components/PositionTable'
import dayjs from 'dayjs'

function PortfolioPage() {
  const [positions, setPositions] = useState<Position[]>([])
  const [performance, setPerformance] = useState<PortfolioPerformance | null>(null)
  const [trades, setTrades] = useState<PortfolioTrade[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPositions = useCallback(async () => {
    try {
      const response = await client.get<ApiResponse<Position[]>>('/portfolio/positions')
      if (response.data.success) {
        setPositions(response.data.data)
      }
    } catch (err) {
      console.error('Fetch positions error:', err)
      setError('获取持仓数据失败')
    }
  }, [])

  const fetchPerformance = useCallback(async () => {
    try {
      const response = await client.get<ApiResponse<PortfolioPerformance>>('/portfolio/performance')
      if (response.data.success) {
        setPerformance(response.data.data)
      }
    } catch (err) {
      console.error('Fetch performance error:', err)
    }
  }, [])

  const fetchTrades = useCallback(async () => {
    try {
      const response = await client.get<ApiResponse<PortfolioTrade[]>>('/portfolio/trades')
      if (response.data.success) {
        setTrades(response.data.data)
      }
    } catch (err) {
      console.error('Fetch trades error:', err)
    }
  }, [])

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchPositions(), fetchPerformance(), fetchTrades()])
      setLoading(false)
    }
    loadData()
  }, [fetchPositions, fetchPerformance, fetchTrades])

  const navChartOption = {
    backgroundColor: 'transparent',
    title: {
      text: '模拟盘净值曲线',
      left: 'center',
      textStyle: { color: '#e5e7eb', fontSize: 14 },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1f2937',
      borderColor: '#374151',
      textStyle: { color: '#e5e7eb' },
    },
    grid: {
      left: '3%',
      right: '3%',
      bottom: '12%',
      top: '15%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: performance?.nav_history.map((p) => p.date) || [],
      axisLine: { lineStyle: { color: '#4b5563' } },
      axisLabel: { color: '#9ca3af', fontSize: 10, rotate: 30 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#4b5563' } },
      axisLabel: { color: '#9ca3af', fontSize: 10 },
      splitLine: { lineStyle: { color: '#374151', type: 'dashed' } },
    },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      {
        type: 'slider',
        start: 0,
        end: 100,
        height: 20,
        bottom: 5,
        borderColor: '#4b5563',
        backgroundColor: '#1f2937',
        fillerColor: 'rgba(59,130,246,0.2)',
        textStyle: { color: '#9ca3af' },
      },
    ],
    series: [
      {
        name: '净值',
        type: 'line',
        data: performance?.nav_history.map((p) => [p.date, p.value]) || [],
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#8b5cf6' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(139,92,246,0.3)' },
              { offset: 1, color: 'rgba(139,92,246,0)' },
            ],
          },
        },
      },
    ],
  }

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-800 rounded w-48" />
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 bg-gray-800 rounded-lg" />
            ))}
          </div>
          <div className="h-64 bg-gray-800 rounded-lg" />
          <div className="h-48 bg-gray-800 rounded-lg" />
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-bold text-white">模拟盘</h1>

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

      {/* Overview Cards */}
      {performance && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="text-xs text-gray-400 mb-1">总资产</div>
            <div className="text-lg font-bold text-white">
              {performance.total_value.toLocaleString('zh-CN', { style: 'currency', currency: 'CNY' })}
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="text-xs text-gray-400 mb-1">总盈亏</div>
            <div className={`text-lg font-bold ${performance.total_pnl >= 0 ? 'text-red-400' : 'text-green-400'}`}>
              {performance.total_pnl >= 0 ? '+' : ''}
              {performance.total_pnl.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
            </div>
            <div className={`text-xs ${performance.total_pnl_pct >= 0 ? 'text-red-400' : 'text-green-400'}`}>
              {performance.total_pnl_pct >= 0 ? '+' : ''}
              {(performance.total_pnl_pct * 100).toFixed(2)}%
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="text-xs text-gray-400 mb-1">持仓市值</div>
            <div className="text-lg font-bold text-blue-400">
              {performance.positions_value.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="text-xs text-gray-400 mb-1">可用现金</div>
            <div className="text-lg font-bold text-gray-300">
              {performance.cash.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
            </div>
          </div>
        </div>
      )}

      {/* Nav Chart */}
      {performance && performance.nav_history.length > 0 ? (
        <div className="bg-gray-800 rounded-lg p-4">
          <ReactECharts
            option={navChartOption}
            style={{ height: '300px', width: '100%' }}
            opts={{ renderer: 'canvas' }}
            notMerge={true}
          />
        </div>
      ) : (
        <div className="flex items-center justify-center h-64 bg-gray-800 rounded-lg">
          <div className="text-gray-500 text-sm">暂无净值数据</div>
        </div>
      )}

      {/* Positions */}
      <PositionTable positions={positions} loading={false} />

      {/* Trade History */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-sm font-semibold text-white">交易历史</h3>
          <p className="text-xs text-gray-400 mt-1">共 {trades.length} 笔交易</p>
        </div>
        {trades.length > 0 ? (
          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-800">
                <tr className="text-xs text-gray-400 border-b border-gray-700">
                  <th className="text-left px-4 py-3 font-medium">日期</th>
                  <th className="text-left px-4 py-3 font-medium">ETF</th>
                  <th className="text-center px-4 py-3 font-medium">方向</th>
                  <th className="text-right px-4 py-3 font-medium">价格</th>
                  <th className="text-right px-4 py-3 font-medium">数量</th>
                  <th className="text-right px-4 py-3 font-medium">金额</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade) => (
                  <tr
                    key={trade.id}
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500 text-sm">暂无交易记录</div>
        )}
      </div>
    </div>
  )
}

export default PortfolioPage
