import { useState } from 'react'

interface StrategyDetail {
  id: string
  name: string
  subtitle: string
  sharpeIS: string
  sharpeOOS: string
  annualReturn: string
  maxDD: string
  tradingFreq: string
  components: { name: string; contribution: string; logic: string }[]
  parameters: { name: string; value: string; description: string }[]
  suitable: string
}

const strategies: StrategyDetail[] = [
  {
    id: 'hunter',
    name: '猎手模式',
    subtitle: '激进追涨 · 快速重入',
    sharpeIS: '4.2',
    sharpeOOS: '7.64',
    annualReturn: '~45%',
    maxDD: '~6.5%',
    tradingFreq: '~2.1次/周',
    suitable: '适合高频关注、追求极致收益的用户。市场趋势明确时表现最优。',
    components: [
      {
        name: 'Composite 信号层',
        contribution: '信号质量提升 ~40%（vs 单一动量）',
        logic: '4个独立子信号（Holt趋势 + EWMA多尺度 + SavGol平滑斜率 + 动量质量评分）各自z-normalize后取均值。多信号融合降低单因子噪声，提供更稳定的横截面排序。',
      },
      {
        name: '渐进式 ATR 跟踪止损',
        contribution: '回撤削减 60%（从 ~15% → ~6%）',
        logic: '新仓用紧止损(0.8x ATR)快速认错；盈利2%后放宽至1.2x ATR让利润跑；盈利5%后进一步放宽至1.8x ATR保护大赢。形成非对称收益分布：小亏大赚。',
      },
      {
        name: '硬止损 -2%',
        contribution: '尾部风险兜底',
        logic: '无论ATR信号如何，入场价-2%直接平仓。防止ATR在极端跳空时失效。',
      },
      {
        name: '滚动回撤制动',
        contribution: 'Sharpe 提升 ~0.3，避免连亏螺旋',
        logic: '3日累计亏损>2%暂停3天，5日>3%暂停5天。30天内反复触发则递进延长。打破\"越亏越急\"的行为陷阱。',
      },
      {
        name: 'VR(5) 方差比率加权',
        contribution: 'Sharpe 提升 ~0.15',
        logic: 'VR(5)=Var(5日收益)/(5×Var(1日收益))。VR>1表示趋势环境（动量有效），加权；VR<1表示震荡（动量失效），降权。只在有效环境下加大仓位。',
      },
      {
        name: '止损后立即重入',
        contribution: '年化提升 ~8%（vs等待下周期）',
        logic: 'Hunter特有：止损触发后如果有新的强势信号则立即建仓，不浪费等待时间。代价是交易频率略高，但在趋势行情中获得更好的时间暴露。',
      },
      {
        name: '极端 Regime 过滤',
        contribution: '灾难保护（极端熊市避免 -20%+）',
        logic: '仅在沪深300<MA20<MA60且5日跌>5%时清仓。设计为\"保险\"而非常规择时，避免频繁误判。',
      },
    ],
    parameters: [
      { name: '再平衡周期', value: '5天', description: '固定5个交易日重新计算持仓' },
      { name: '持仓数', value: 'Top 2', description: '集中持有2只最强ETF' },
      { name: 'ATR倍数', value: '0.8 / 1.2 / 1.8', description: '新仓 / 盈利2%+ / 盈利5%+' },
      { name: '硬止损', value: '-2%', description: '入场价下跌2%直接切' },
      { name: '加权方式', value: '反波动率', description: '波动率越低分配越多（稳定性偏好）' },
      { name: 'DD制动', value: '-2%/3d, -3%/5d', description: '3日累计-2%暂停3天，5日累计-3%暂停5天' },
    ],
  },
  {
    id: 'steady',
    name: '稳健模式',
    subtitle: '低频稳健 · 排除重入',
    sharpeIS: '3.8',
    sharpeOOS: '6.94',
    annualReturn: '~43%',
    maxDD: '~7%',
    tradingFreq: '~1.6次/周',
    suitable: '适合不想频繁关注、偏好低交易频率的用户。更"set and forget"。',
    components: [
      {
        name: 'Composite 信号层',
        contribution: '同猎手模式',
        logic: '4个独立子信号z-normalize后取均值（Holt + EWMA + SavGol + MomQuality）。',
      },
      {
        name: '渐进式 ATR 跟踪止损',
        contribution: '同猎手模式',
        logic: '0.8x / 1.2x / 1.8x 三档渐进，配合-2%硬止损。',
      },
      {
        name: '硬止损 -2%',
        contribution: '同猎手模式',
        logic: '绝对底线保护。',
      },
      {
        name: '滚动回撤制动',
        contribution: '同猎手模式',
        logic: '3日/5日累计亏损制动 + 递进暂停。',
      },
      {
        name: 'VR(5) 方差比率加权',
        contribution: '同猎手模式',
        logic: '趋势环境加权，震荡降权。',
      },
      {
        name: '止损ETF排除机制',
        contribution: '交易频率降低 ~25%',
        logic: 'Steady特有：止损后该ETF在本再平衡周期内不允许被重新选入。等下个周期开始时才清除黑名单。避免"止损→立即买回→再止损"的死循环。',
      },
      {
        name: '极端 Regime 过滤',
        contribution: '同猎手模式',
        logic: '仅极端熊市清仓。',
      },
    ],
    parameters: [
      { name: '再平衡周期', value: '7天', description: '固定7个交易日重新计算持仓' },
      { name: '持仓数', value: 'Top 2', description: '集中持有2只最强ETF' },
      { name: 'ATR倍数', value: '0.8 / 1.2 / 1.8', description: '新仓 / 盈利2%+ / 盈利5%+' },
      { name: '硬止损', value: '-2%', description: '入场价下跌2%直接切' },
      { name: '加权方式', value: '反波动率', description: '波动率越低分配越多' },
      { name: 'DD制动', value: '-2%/3d, -3%/5d', description: '3日累计-2%暂停3天，5日累计-3%暂停5天' },
      { name: '排除规则', value: '本周期内', description: '止损ETF在当前7天周期内不重入' },
    ],
  },
]

function StrategyPage() {
  const [expanded, setExpanded] = useState<string>('hunter')

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">策略管理</h1>
        <p className="text-sm text-gray-400 mt-1">
          查看策略详细逻辑与每个组件的贡献。在回测页面可选择策略运行。
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {strategies.map(s => (
          <div
            key={s.id}
            className={`p-5 rounded-xl border cursor-pointer transition-all ${
              expanded === s.id
                ? 'border-blue-500 bg-blue-500/5 shadow-lg shadow-blue-500/10'
                : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
            }`}
            onClick={() => setExpanded(expanded === s.id ? '' : s.id)}
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">{s.name}</h3>
                <p className="text-xs text-gray-400 mt-0.5">{s.subtitle}</p>
              </div>
              <span className={`text-xs px-2 py-0.5 rounded ${
                s.id === 'hunter' ? 'bg-orange-500/20 text-orange-400' : 'bg-green-500/20 text-green-400'
              }`}>
                {s.id === 'hunter' ? '激进' : '稳健'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 mt-4 text-sm">
              <div>
                <span className="text-gray-500">OOS Sharpe</span>
                <span className="float-right text-blue-400 font-medium">{s.sharpeOOS}</span>
              </div>
              <div>
                <span className="text-gray-500">年化收益</span>
                <span className="float-right text-green-400 font-medium">{s.annualReturn}</span>
              </div>
              <div>
                <span className="text-gray-500">最大回撤</span>
                <span className="float-right text-red-400 font-medium">{s.maxDD}</span>
              </div>
              <div>
                <span className="text-gray-500">交易频率</span>
                <span className="float-right text-gray-300">{s.tradingFreq}</span>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-3 italic">{s.suitable}</p>
          </div>
        ))}
      </div>

      {/* Detailed Breakdown */}
      {expanded && (() => {
        const s = strategies.find(x => x.id === expanded)!
        return (
          <div className="space-y-6">
            {/* Components */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
              <h3 className="text-base font-semibold text-white mb-4">
                组件分解 — {s.name}
              </h3>
              <div className="space-y-4">
                {s.components.map((c, i) => (
                  <div key={i} className="border-l-2 border-blue-500/40 pl-4">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-white">{c.name}</span>
                      <span className="text-xs text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                        {c.contribution}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 mt-1 leading-relaxed">{c.logic}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Parameters */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
              <h3 className="text-base font-semibold text-white mb-4">参数设置</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b border-gray-700">
                    <th className="pb-2 font-normal">参数</th>
                    <th className="pb-2 font-normal">值</th>
                    <th className="pb-2 font-normal">说明</th>
                  </tr>
                </thead>
                <tbody>
                  {s.parameters.map((p, i) => (
                    <tr key={i} className="border-b border-gray-700/50">
                      <td className="py-2 text-gray-300">{p.name}</td>
                      <td className="py-2 text-blue-400 font-mono">{p.value}</td>
                      <td className="py-2 text-gray-500">{p.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Comparison */}
            {expanded === 'hunter' && (
              <div className="bg-yellow-500/5 border border-yellow-600/30 rounded-xl p-5">
                <h4 className="text-sm font-medium text-yellow-400 mb-2">与稳健模式的核心区别</h4>
                <ul className="text-xs text-gray-400 space-y-1.5 list-disc list-inside">
                  <li>再平衡周期更短 (5天 vs 7天) → 更敏捷抓住趋势变化</li>
                  <li>止损后立即重入 → 不浪费有效趋势暴露时间</li>
                  <li>交易频率更高 (~2.1次/周 vs ~1.6次/周) → 需要更频繁关注</li>
                  <li>在强趋势行情中收益更高；但震荡市可能多付交易成本</li>
                </ul>
              </div>
            )}
            {expanded === 'steady' && (
              <div className="bg-green-500/5 border border-green-600/30 rounded-xl p-5">
                <h4 className="text-sm font-medium text-green-400 mb-2">与猎手模式的核心区别</h4>
                <ul className="text-xs text-gray-400 space-y-1.5 list-disc list-inside">
                  <li>再平衡周期更长 (7天 vs 5天) → 更稳定，降低"来回跳"</li>
                  <li>止损后排除该ETF等待下周期 → 避免"止损→重买→再止损"循环</li>
                  <li>交易频率更低 (~1.6次/周 vs ~2.1次/周) → 更少人工干预</li>
                  <li>在震荡市节省交易成本；但强趋势中可能错过短期重入机会</li>
                </ul>
              </div>
            )}
          </div>
        )
      })()}

      {/* Methodology Note */}
      <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-5">
        <h4 className="text-sm font-medium text-gray-300 mb-2">验证方法论</h4>
        <div className="text-xs text-gray-500 space-y-1">
          <p>样本内 (In-Sample): 2020-01 ~ 2025-05，约5.4年</p>
          <p>样本外 (Out-of-Sample): 2026-01 ~ 2026-05，未参与任何参数调优</p>
          <p>费率: 万5 双边 (每买+每卖各万2.5)，包含在所有回测收益中</p>
          <p>参数稳定性: 对所有关键参数做 ±30% 敏感度测试，确认非过拟合</p>
        </div>
      </div>
    </div>
  )
}

export default StrategyPage
