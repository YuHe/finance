import ReactECharts from 'echarts-for-react'
import type { NavPoint } from '../api/client'

interface NavChartProps {
  strategyNav: NavPoint[]
  benchmarkNav: NavPoint[]
  loading?: boolean
}

function NavChart({ strategyNav, benchmarkNav, loading }: NavChartProps) {
  const option = {
    backgroundColor: 'transparent',
    title: {
      text: '策略净值 vs 沪深300基准',
      left: 'center',
      textStyle: {
        color: '#e5e7eb',
        fontSize: 14,
      },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1f2937',
      borderColor: '#374151',
      textStyle: { color: '#e5e7eb' },
      formatter: (params: Array<{ seriesName: string; value: [string, number]; color: string }>) => {
        if (!params.length) return ''
        let html = `<div class="text-xs">${params[0].value[0]}<br/>`
        params.forEach((p) => {
          html += `<span style="color:${p.color}">${p.seriesName}: ${p.value[1].toFixed(4)}</span><br/>`
        })
        html += '</div>'
        return html
      },
    },
    legend: {
      top: 30,
      textStyle: { color: '#9ca3af' },
      data: ['策略净值', '沪深300'],
    },
    grid: {
      left: '3%',
      right: '3%',
      bottom: '12%',
      top: '18%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: strategyNav.map((p) => p.date),
      axisLine: { lineStyle: { color: '#4b5563' } },
      axisLabel: {
        color: '#9ca3af',
        fontSize: 10,
        rotate: 30,
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#4b5563' } },
      axisLabel: { color: '#9ca3af', fontSize: 10 },
      splitLine: { lineStyle: { color: '#374151', type: 'dashed' } },
    },
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100,
      },
      {
        type: 'slider',
        start: 0,
        end: 100,
        height: 20,
        bottom: 5,
        borderColor: '#4b5563',
        backgroundColor: '#1f2937',
        fillerColor: 'rgba(59,130,246,0.2)',
        handleStyle: { color: '#3b82f6' },
        textStyle: { color: '#9ca3af' },
      },
    ],
    series: [
      {
        name: '策略净值',
        type: 'line',
        data: strategyNav.map((p) => [p.date, p.value]),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#3b82f6' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(59,130,246,0.3)' },
              { offset: 1, color: 'rgba(59,130,246,0)' },
            ],
          },
        },
      },
      {
        name: '沪深300',
        type: 'line',
        data: benchmarkNav.map((p) => [p.date, p.value]),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#f59e0b' },
      },
    ],
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-80 bg-gray-800 rounded-lg">
        <div className="text-gray-400 text-sm">加载中...</div>
      </div>
    )
  }

  if (!strategyNav.length) {
    return (
      <div className="flex items-center justify-center h-80 bg-gray-800 rounded-lg">
        <div className="text-gray-500 text-sm">暂无数据，请先运行回测</div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <ReactECharts
        option={option}
        style={{ height: '360px', width: '100%' }}
        opts={{ renderer: 'canvas' }}
        notMerge={true}
      />
    </div>
  )
}

export default NavChart
