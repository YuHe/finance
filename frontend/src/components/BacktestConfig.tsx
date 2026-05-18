import { useState, useEffect } from 'react'
import client from '../api/client'
import type { BacktestParams } from '../api/client'

interface BacktestConfigProps {
  onSubmit: (params: BacktestParams) => void
  loading: boolean
}

interface EtfInfo {
  code: string
  name: string
  industry: string
}

function BacktestConfig({ onSubmit, loading }: BacktestConfigProps) {
  const [params, setParams] = useState<BacktestParams>({
    start_date: '2026-01-01',
    end_date: new Date().toISOString().slice(0, 10),
    initial_capital: 1000000,
    top_n: 1,
    weight_method: 'equal',
    rebalance_freq: 'weekly',
    momentum_window: 20,
    stop_loss_enabled: true,
    stop_loss_threshold: 0.08,
    trailing_stop: false,
    trailing_stop_threshold: 0.05,
    selected_codes: null,
  })

  const [etfList, setEtfList] = useState<EtfInfo[]>([])
  const [selectedCodes, setSelectedCodes] = useState<Set<string>>(new Set())
  const [selectAll, setSelectAll] = useState(true)
  const [showEtfPanel, setShowEtfPanel] = useState(false)

  useEffect(() => {
    client.get<{ etfs: EtfInfo[] }>('/market/etfs').then(res => {
      setEtfList(res.data.etfs)
    }).catch(() => {})
  }, [])

  const handleChange = (field: keyof BacktestParams, value: string | number | boolean) => {
    setParams((prev) => ({ ...prev, [field]: value }))
  }

  const toggleEtf = (code: string) => {
    const next = new Set(selectedCodes)
    if (next.has(code)) next.delete(code)
    else next.add(code)
    setSelectedCodes(next)
    setSelectAll(false)
  }

  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedCodes(new Set())
      setSelectAll(false)
    } else {
      setSelectedCodes(new Set(etfList.map(e => e.code)))
      setSelectAll(true)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const codes = selectAll ? null : (selectedCodes.size > 0 ? Array.from(selectedCodes) : null)
    onSubmit({ ...params, selected_codes: codes })
  }

  const formatCapital = (v: number) => {
    if (v >= 10000) return `${(v / 10000).toFixed(v % 10000 === 0 ? 0 : 1)}万`
    return v.toLocaleString()
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <h2 className="text-lg font-semibold text-white mb-4">回测参数配置</h2>

      {/* Capital */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">
          回测金额: <span className="text-green-400 font-medium">{formatCapital(params.initial_capital)}</span>
        </label>
        <input
          type="number"
          min={10000}
          step={10000}
          value={params.initial_capital}
          onChange={(e) => handleChange('initial_capital', parseFloat(e.target.value) || 100000)}
          className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="100万"
        />
        <div className="flex gap-2 mt-1.5">
          {[100000, 500000, 1000000, 2000000].map(v => (
            <button
              key={v}
              type="button"
              onClick={() => handleChange('initial_capital', v)}
              className={`text-xs px-2 py-0.5 rounded ${
                params.initial_capital === v
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }`}
            >
              {formatCapital(v)}
            </button>
          ))}
        </div>
      </div>

      {/* Date Range */}
      <div className="space-y-2">
        <label className="block text-xs text-gray-400">回测时间</label>
        <div className="flex flex-wrap gap-1.5 mb-2">
          {[
            { label: '今年以来', start: `${new Date().getFullYear()}-01-01` },
            { label: '近1年', start: new Date(Date.now() - 365 * 86400000).toISOString().slice(0, 10) },
            { label: '近3年', start: new Date(Date.now() - 3 * 365 * 86400000).toISOString().slice(0, 10) },
            { label: '近5年', start: new Date(Date.now() - 5 * 365 * 86400000).toISOString().slice(0, 10) },
          ].map(preset => {
            const endToday = new Date().toISOString().slice(0, 10)
            const isActive = params.start_date === preset.start && params.end_date === endToday
            return (
              <button
                key={preset.label}
                type="button"
                onClick={() => {
                  handleChange('start_date', preset.start)
                  handleChange('end_date', endToday)
                }}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-gray-200'
                }`}
              >
                {preset.label}
              </button>
            )
          })}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">起始</label>
            <input
              type="date"
              value={params.start_date}
              onChange={(e) => handleChange('start_date', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">结束</label>
            <input
              type="date"
              value={params.end_date}
              onChange={(e) => handleChange('end_date', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* ETF Selection */}
      <div className="space-y-2 p-3 bg-gray-800 rounded-lg border border-gray-700">
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400">
            参与回测 ETF: <span className="text-blue-400 font-medium">
              {selectAll ? '全部' : `${selectedCodes.size}/${etfList.length}`}
            </span>
          </label>
          <button
            type="button"
            onClick={() => setShowEtfPanel(!showEtfPanel)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {showEtfPanel ? '收起' : '选择'}
          </button>
        </div>
        {showEtfPanel && (
          <div className="space-y-1.5 max-h-48 overflow-y-auto">
            <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer px-1 py-0.5 rounded hover:bg-gray-700">
              <input
                type="checkbox"
                checked={selectAll}
                onChange={handleSelectAll}
                className="w-3.5 h-3.5 accent-blue-500"
              />
              <span className="font-medium">全选 / 全不选</span>
            </label>
            <div className="border-t border-gray-700 my-1" />
            {etfList.map(etf => (
              <label
                key={etf.code}
                className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer px-1 py-0.5 rounded hover:bg-gray-700"
              >
                <input
                  type="checkbox"
                  checked={selectAll || selectedCodes.has(etf.code)}
                  onChange={() => {
                    if (selectAll) {
                      // 从全选切换到取消某一个
                      const all = new Set(etfList.map(e => e.code))
                      all.delete(etf.code)
                      setSelectedCodes(all)
                      setSelectAll(false)
                    } else {
                      toggleEtf(etf.code)
                    }
                  }}
                  className="w-3.5 h-3.5 accent-blue-500"
                />
                <span className="text-gray-500 font-mono">{etf.code}</span>
                <span>{etf.name}</span>
                <span className="text-gray-600 ml-auto">{etf.industry}</span>
              </label>
            ))}
          </div>
        )}
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
