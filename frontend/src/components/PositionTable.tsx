import type { Position } from '../api/client'

interface PositionTableProps {
  positions: Position[]
  loading?: boolean
}

function PositionTable({ positions, loading }: PositionTableProps) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-gray-700 rounded w-32" />
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-10 bg-gray-700 rounded" />
          ))}
        </div>
      </div>
    )
  }

  if (!positions.length) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-500 text-sm">
        暂无持仓
      </div>
    )
  }

  const totalValue = positions.reduce((sum, p) => sum + p.market_value, 0)

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-white">当前持仓</h3>
        <p className="text-xs text-gray-400 mt-1">
          总市值: <span className="text-blue-400 font-medium">{totalValue.toLocaleString('zh-CN', { style: 'currency', currency: 'CNY' })}</span>
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-400 border-b border-gray-700">
              <th className="text-left px-4 py-3 font-medium">ETF</th>
              <th className="text-left px-4 py-3 font-medium">行业</th>
              <th className="text-right px-4 py-3 font-medium">持仓数量</th>
              <th className="text-right px-4 py-3 font-medium">成本价</th>
              <th className="text-right px-4 py-3 font-medium">现价</th>
              <th className="text-right px-4 py-3 font-medium">市值</th>
              <th className="text-right px-4 py-3 font-medium">盈亏</th>
              <th className="text-right px-4 py-3 font-medium">盈亏比例</th>
              <th className="text-right px-4 py-3 font-medium">仓位占比</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos) => (
              <tr
                key={pos.etf_code}
                className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors"
              >
                <td className="px-4 py-3">
                  <div className="font-medium text-white">{pos.etf_name}</div>
                  <div className="text-xs text-gray-500">{pos.etf_code}</div>
                </td>
                <td className="px-4 py-3 text-gray-300">{pos.industry}</td>
                <td className="px-4 py-3 text-right text-gray-300">
                  {pos.shares.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right text-gray-300">
                  {pos.avg_cost.toFixed(3)}
                </td>
                <td className="px-4 py-3 text-right text-white font-medium">
                  {pos.current_price.toFixed(3)}
                </td>
                <td className="px-4 py-3 text-right text-gray-300">
                  {pos.market_value.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
                </td>
                <td className={`px-4 py-3 text-right font-medium ${pos.pnl >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                  {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
                </td>
                <td className={`px-4 py-3 text-right font-medium ${pos.pnl_pct >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                  {pos.pnl_pct >= 0 ? '+' : ''}{(pos.pnl_pct * 100).toFixed(2)}%
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="flex items-center justify-end gap-2">
                    <div className="w-16 h-1.5 bg-gray-700 rounded-full">
                      <div
                        className="h-full bg-blue-500 rounded-full"
                        style={{ width: `${pos.weight * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400">
                      {(pos.weight * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default PositionTable
