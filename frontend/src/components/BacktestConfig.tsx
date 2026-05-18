import { useState } from 'react'
import type { BacktestParams } from '../api/client'

interface BacktestConfigProps {
  onSubmit: (params: BacktestParams) => void
  loading: boolean
}

function BacktestConfig({ onSubmit, loading }: BacktestConfigProps) {
  const [params, setParams] = useState<BacktestParams>({
    start_date: '2020-01-01',
    end_date: '2024-12-31',
    top_n: 3,
    weight_method: 'equal',
    rebalance_freq: 'weekly',
    momentum_window: 20,
    stop_loss_enabled: true,
    stop_loss_threshold: 0.08,
    trailing_stop: false,
    trailing_stop_threshold: 0.05,
  })

  const handleChange = (field: keyof BacktestParams, value: string | number | boolean) => {
    setParams((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(params)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <h2 className="text-lg font-semibold text-white mb-4">回测参数配置</h2>

      {/* Date Range */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-400 mb-1">开始日期</label>
          <input
            type="date"
            value={params.start_date}
            onChange={(e) => handleChange('start_date', e.target.value)}
            className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">结束日期</label>
          <input
            type="date"
            value={params.end_date}
            onChange={(e) => handleChange('end_date', e.target.value)}
            className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Top N */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">
          持仓数量 (Top N): <span className="text-blue-400 font-medium">{params.top_n}</span>
        </label>
        <input
          type="range"
          min={1}
          max={5}
          value={params.top_n}
          onChange={(e) => handleChange('top_n', parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>2</span>
          <span>3</span>
          <span>4</span>
          <span>5</span>
        </div>
      </div>

      {/* Weight Method */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">权重分配方式</label>
        <select
          value={params.weight_method}
          onChange={(e) => handleChange('weight_method', e.target.value)}
          className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="equal">等权重</option>
          <option value="momentum_weighted">动量加权</option>
          <option value="inverse_volatility">波动率倒数</option>
        </select>
      </div>

      {/* Rebalance Frequency */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">调仓频率</label>
        <select
          value={params.rebalance_freq}
          onChange={(e) => handleChange('rebalance_freq', e.target.value)}
          className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="weekly">每周</option>
          <option value="biweekly">双周</option>
          <option value="monthly">每月</option>
        </select>
      </div>

      {/* Momentum Window */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">
          动量窗口 (交易日): <span className="text-blue-400 font-medium">{params.momentum_window}</span>
        </label>
        <input
          type="range"
          min={5}
          max={60}
          step={5}
          value={params.momentum_window}
          onChange={(e) => handleChange('momentum_window', parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>5</span>
          <span>20</span>
          <span>40</span>
          <span>60</span>
        </div>
      </div>

      {/* Stop Loss */}
      <div className="space-y-3 p-3 bg-gray-800 rounded-lg border border-gray-700">
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400">启用止损</label>
          <input
            type="checkbox"
            checked={params.stop_loss_enabled}
            onChange={(e) => handleChange('stop_loss_enabled', e.target.checked)}
            className="w-4 h-4 accent-blue-500"
          />
        </div>
        {params.stop_loss_enabled && (
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              止损阈值: <span className="text-red-400">{(params.stop_loss_threshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min={0.03}
              max={0.2}
              step={0.01}
              value={params.stop_loss_threshold}
              onChange={(e) => handleChange('stop_loss_threshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-red-500"
            />
          </div>
        )}

        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400">移动止盈</label>
          <input
            type="checkbox"
            checked={params.trailing_stop}
            onChange={(e) => handleChange('trailing_stop', e.target.checked)}
            className="w-4 h-4 accent-blue-500"
          />
        </div>
        {params.trailing_stop && (
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              回撤阈值: <span className="text-yellow-400">{(params.trailing_stop_threshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min={0.03}
              max={0.15}
              step={0.01}
              value={params.trailing_stop_threshold}
              onChange={(e) => handleChange('trailing_stop_threshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-yellow-500"
            />
          </div>
        )}
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>回测运行中...</span>
          </>
        ) : (
          <span>开始回测</span>
        )}
      </button>
    </form>
  )
}

export default BacktestConfig
