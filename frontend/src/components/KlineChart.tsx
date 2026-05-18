import ReactECharts from 'echarts-for-react'
import type { TradeRecord } from '../api/client'

interface KlineChartProps {
  klineData: Array<{
    date: string
    open: number
    close: number
    low: number
    high: number
    volume: number
  }>
  trades: TradeRecord[]
  title?: string
  loading?: boolean
}

function KlineChart({ klineData, trades, title = 'K线图', loading }: KlineChartProps) {
  const dates = klineData.map((d) => d.date)
  const ohlc = klineData.map((d) => [d.open, d.close, d.low, d.high])
  const volumes = klineData.map((d) => d.volume)

  // Create buy/sell markers from trades
  const buyMarkers = trades
    .filter((t) => t.direction === 'buy')
    .map((t) => {
      const idx = dates.indexOf(t.date)
      if (idx === -1) return null
      return {
        name: '买入',
        coord: [t.date, klineData[idx]?.low ?? t.price],
        value: t.etf_name,
        itemStyle: { color: '#ef4444' },
        symbol: 'triangle',
        symbolSize: 12,
      }
    })
    .filter(Boolean)

  const sellMarkers = trades
    .filter((t) => t.direction === 'sell')
    .map((t) => {
      const idx = dates.indexOf(t.date)
      if (idx === -1) return null
      return {
        name: '卖出',
        coord: [t.date, klineData[idx]?.high ?? t.price],
        value: t.etf_name,
        itemStyle: { color: '#22c55e' },
        symbol: 'pin',
        symbolSize: 12,
        symbolRotate: 180,
      }
    })
    .filter(Boolean)

  const option = {
    backgroundColor: 'transparent',
    title: {
      text: title,
      left: 'center',
      textStyle: { color: '#e5e7eb', fontSize: 14 },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: '#1f2937',
      borderColor: '#374151',
      textStyle: { color: '#e5e7eb' },
    },
    legend: {
      top: 30,
      textStyle: { color: '#9ca3af' },
    },
    grid: [
      { left: '8%', right: '3%', top: '15%', height: '55%' },
      { left: '8%', right: '3%', top: '75%', height: '15%' },
    ],
    xAxis: [
      {
        type: 'category',
        data: dates,
        gridIndex: 0,
        axisLine: { lineStyle: { color: '#4b5563' } },
        axisLabel: { color: '#9ca3af', fontSize: 10 },
        splitLine: { show: false },
      },
      {
        type: 'category',
        data: dates,
        gridIndex: 1,
        axisLine: { lineStyle: { color: '#4b5563' } },
        axisLabel: { show: false },
        splitLine: { show: false },
      },
    ],
    yAxis: [
      {
        type: 'value',
        gridIndex: 0,
        axisLine: { lineStyle: { color: '#4b5563' } },
        axisLabel: { color: '#9ca3af', fontSize: 10 },
        splitLine: { lineStyle: { color: '#374151', type: 'dashed' } },
      },
      {
        type: 'value',
        gridIndex: 1,
        axisLine: { lineStyle: { color: '#4b5563' } },
        axisLabel: { show: false },
        splitLine: { show: false },
      },
    ],
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: [0, 1],
        start: 70,
        end: 100,
      },
      {
        type: 'slider',
        xAxisIndex: [0, 1],
        start: 70,
        end: 100,
        height: 15,
        bottom: 2,
        borderColor: '#4b5563',
        backgroundColor: '#1f2937',
        fillerColor: 'rgba(59,130,246,0.2)',
        textStyle: { color: '#9ca3af' },
      },
    ],
    series: [
      {
        name: 'K线',
        type: 'candlestick',
        data: ohlc,
        xAxisIndex: 0,
        yAxisIndex: 0,
        itemStyle: {
          color: '#ef4444',
          color0: '#22c55e',
          borderColor: '#ef4444',
          borderColor0: '#22c55e',
        },
        markPoint: {
          data: [...buyMarkers, ...sellMarkers],
          label: {
            show: true,
            formatter: (p: { name: string }) => p.name,
            fontSize: 9,
            color: '#fff',
          },
        },
      },
      {
        name: '成交量',
        type: 'bar',
        data: volumes,
        xAxisIndex: 1,
        yAxisIndex: 1,
        itemStyle: {
          color: (params: { dataIndex: number }) => {
            const idx = params.dataIndex
            if (idx > 0) {
              return klineData[idx].close >= klineData[idx - 1].close ? '#ef4444' : '#22c55e'
            }
            return '#6b7280'
          },
        },
      },
    ],
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-800 rounded-lg">
        <div className="text-gray-400 text-sm">加载中...</div>
      </div>
    )
  }

  if (!klineData.length) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-800 rounded-lg">
        <div className="text-gray-500 text-sm">暂无K线数据</div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <ReactECharts
        option={option}
        style={{ height: '420px', width: '100%' }}
        opts={{ renderer: 'canvas' }}
        notMerge={true}
      />
    </div>
  )
}

export default KlineChart
