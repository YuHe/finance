import { useState, useEffect } from 'react'
import KlineChart from './KlineChart'
import { fetchKlineData } from '../api/client'
import type { KlineDataPoint, TradeRecord } from '../api/client'

interface KlineModalProps {
  code: string
  name: string
  trades?: TradeRecord[]
  onClose: () => void
}

function KlineModal({ code, name, trades = [], onClose }: KlineModalProps) {
  const [klineData, setKlineData] = useState<KlineDataPoint[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchKlineData(code)
      .then((data) => {
        if (!cancelled) setKlineData(data)
      })
      .catch((err) => {
        console.error('Failed to fetch kline:', err)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [code])

  // Filter trades for this specific ETF
  const etfTrades = trades.filter((t) => t.etf_code === code)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-xl border border-gray-600 shadow-2xl w-[90vw] max-w-5xl max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-white">{name}</h2>
            <span className="text-sm text-gray-400">{code}</span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl leading-none px-2"
          >
            &times;
          </button>
        </div>

        {/* Chart */}
        <div className="flex-1 overflow-hidden p-4">
          <KlineChart
            klineData={klineData}
            trades={etfTrades}
            title={`${name} (${code})`}
            loading={loading}
          />
        </div>
      </div>
    </div>
  )
}

export default KlineModal
