import { useState, useEffect } from 'react'
import client from '../api/client'
import KlineModal from '../components/KlineModal'

interface EtfDataStatus {
  code: string
  name: string
  industry: string
  start_date: string | null
  end_date: string | null
  count: number
  has_data: boolean
}

function DataPage() {
  const [dataStatus, setDataStatus] = useState<EtfDataStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [klineTarget, setKlineTarget] = useState<{ code: string; name: string } | null>(null)

  const fetchStatus = async () => {
    try {
      const res = await client.get<{ data: EtfDataStatus[] }>('/market/data-status')
      setDataStatus(res.data.data)
    } catch (e) {
      console.error('Failed to fetch data status', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStatus() }, [])

  // 找出所有有数据的ETF的公共日期范围
  const withData = dataStatus.filter(d => d.has_data)
  const allStart = withData.length > 0
    ? withData.reduce((latest, d) => d.start_date && d.start_date > latest ? d.start_date : latest, '1900-01-01')
    : null
  const allEnd = withData.length > 0
    ? withData.reduce((earliest, d) => d.end_date && d.end_date < earliest ? d.end_date : earliest, '2099-12-31')
    : null

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">加载中...</div>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h1 className="text-xl font-bold text-white mb-2">数据管理</h1>
      <p className="text-sm text-gray-400 mb-6">查看各 ETF 的数据覆盖范围，确定可回测时间段</p>

      {/* 汇总信息 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="text-xs text-gray-400 mb-1">有数据标的数</div>
          <div className="text-2xl font-bold text-green-400">
            {withData.length} <span className="text-sm text-gray-500">/ {dataStatus.length}</span>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="text-xs text-gray-400 mb-1">公共数据起始日</div>
          <div className="text-lg font-bold text-blue-400">
            {allStart && allStart !== '1900-01-01' ? allStart : '—'}
          </div>
          <div className="text-xs text-gray-500 mt-1">所有有数据ETF的最晚起始日</div>
        </div>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="text-xs text-gray-400 mb-1">公共数据结束日</div>
          <div className="text-lg font-bold text-blue-400">
            {allEnd && allEnd !== '2099-12-31' ? allEnd : '—'}
          </div>
          <div className="text-xs text-gray-500 mt-1">所有有数据ETF的最早结束日</div>
        </div>
      </div>

      {/* 刷新按钮 */}
      <div className="flex justify-end mb-3">
        <button
          onClick={() => { setLoading(true); fetchStatus() }}
          className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-3 py-1.5 rounded-lg transition-colors"
        >
          刷新数据状态
        </button>
      </div>

      {/* 数据表格 */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 bg-gray-750">
              <th className="text-left px-4 py-3 text-xs text-gray-400 font-medium">代码</th>
              <th className="text-left px-4 py-3 text-xs text-gray-400 font-medium">名称</th>
              <th className="text-left px-4 py-3 text-xs text-gray-400 font-medium">行业</th>
              <th className="text-left px-4 py-3 text-xs text-gray-400 font-medium">数据起始</th>
              <th className="text-left px-4 py-3 text-xs text-gray-400 font-medium">数据结束</th>
              <th className="text-right px-4 py-3 text-xs text-gray-400 font-medium">交易日数</th>
              <th className="text-center px-4 py-3 text-xs text-gray-400 font-medium">状态</th>
            </tr>
          </thead>
          <tbody>
            {dataStatus.map((item) => (
              <tr
                key={item.code}
                className={`border-b border-gray-700/50 ${
                  item.has_data ? '' : 'opacity-50'
                }`}
              >
                <td className="px-4 py-2.5 text-gray-300 font-mono text-xs">{item.code}</td>
                <td className="px-4 py-2.5">
                  <button
                    className="text-blue-400 hover:underline text-left"
                    onClick={() => item.has_data && setKlineTarget({ code: item.code, name: item.name })}
                    disabled={!item.has_data}
                  >
                    {item.name}
                  </button>
                </td>
                <td className="px-4 py-2.5 text-gray-400">{item.industry}</td>
                <td className="px-4 py-2.5 text-gray-300 font-mono text-xs">
                  {item.start_date || '—'}
                </td>
                <td className="px-4 py-2.5 text-gray-300 font-mono text-xs">
                  {item.end_date || '—'}
                </td>
                <td className="px-4 py-2.5 text-right text-gray-300">
                  {item.count > 0 ? item.count.toLocaleString() : '—'}
                </td>
                <td className="px-4 py-2.5 text-center">
                  {item.has_data ? (
                    <span className="inline-block w-2 h-2 rounded-full bg-green-400" title="有数据" />
                  ) : (
                    <span className="inline-block w-2 h-2 rounded-full bg-red-400" title="无数据" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-gray-500 mt-4">
        提示：回测时间范围应在所选 ETF 的公共数据覆盖区间内。可在回测页面勾选参与回测的 ETF 并设置金额。
      </p>

      {/* Kline Modal */}
      {klineTarget && (
        <KlineModal
          code={klineTarget.code}
          name={klineTarget.name}
          onClose={() => setKlineTarget(null)}
        />
      )}
    </div>
  )
}

export default DataPage
