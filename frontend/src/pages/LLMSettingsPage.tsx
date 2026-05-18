import { useState, useEffect } from 'react'
import client from '../api/client'

interface Provider {
  id: string
  name: string
  api_base: string
  models: { id: string; name: string; supports_search: boolean }[]
  supports_web_search: boolean
}

interface LLMConfig {
  id: number
  provider: string
  api_base: string | null
  model_name: string
  web_search_enabled: boolean
  is_active: boolean
  has_api_key: boolean
}

interface SentimentItem {
  code: string
  score: number
  raw_text: string
  model_used: string
  date: string
}

function LLMSettingsPage() {
  const [providers, setProviders] = useState<Provider[]>([])
  const [config, setConfig] = useState<LLMConfig | null>(null)

  // Form state
  const [provider, setProvider] = useState('dashscope')
  const [apiKey, setApiKey] = useState('')
  const [apiBase, setApiBase] = useState('')
  const [modelName, setModelName] = useState('')
  const [webSearch, setWebSearch] = useState(false)

  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [sentiments, setSentiments] = useState<SentimentItem[]>([])

  useEffect(() => {
    // 加载提供商列表和已有配置
    client.get('/llm/providers').then(res => setProviders(res.data))
    client.get('/llm/config').then(res => {
      if (res.data) {
        setConfig(res.data)
        setProvider(res.data.provider)
        setApiBase(res.data.api_base || '')
        setModelName(res.data.model_name)
        setWebSearch(res.data.web_search_enabled)
      }
    })
    // 加载已有情绪数据
    client.get('/llm/sentiment').then(res => {
      if (res.data?.data) setSentiments(res.data.data)
    }).catch(() => {})
  }, [])

  const currentProvider = providers.find(p => p.id === provider)
  const isCustom = provider === 'custom'
  const availableModels = currentProvider?.models || []
  const supportsSearch = isCustom ? true : (currentProvider?.supports_web_search ?? false)

  // 选择提供商时自动设置默认模型
  const handleProviderChange = (pid: string) => {
    setProvider(pid)
    setTestResult(null)
    if (pid === 'custom') {
      setModelName('')
      setApiBase('')
    } else {
      const p = providers.find(x => x.id === pid)
      if (p) {
        setApiBase('')
        setModelName(p.models[0]?.id || '')
        setWebSearch(p.supports_web_search)
      }
    }
  }

  const handleSave = async () => {
    if (!apiKey && !config?.has_api_key) {
      setTestResult({ success: false, message: '请输入 API Key' })
      return
    }
    setSaving(true)
    try {
      const payload: Record<string, unknown> = {
        provider,
        api_key: apiKey || '__KEEP__',
        model_name: modelName,
        web_search_enabled: webSearch,
      }
      if (isCustom || apiBase) payload.api_base = apiBase
      await client.put('/llm/config', payload)
      setTestResult({ success: true, message: '保存成功' })
      // 刷新配置
      const res = await client.get('/llm/config')
      if (res.data) setConfig(res.data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { error?: { message?: string } } } })?.response?.data?.error?.message || '保存失败'
      setTestResult({ success: false, message: msg })
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await client.post('/llm/test')
      setTestResult({
        success: res.data.success,
        message: res.data.success ? res.data.data.message : res.data.error?.message || '连接失败',
      })
    } catch {
      setTestResult({ success: false, message: '请求失败，请先保存配置' })
    } finally {
      setTesting(false)
    }
  }

  const handleAnalyze = async () => {
    setAnalyzing(true)
    try {
      const res = await client.post('/llm/analyze')
      if (res.data?.data?.results) {
        setSentiments(res.data.data.results)
      }
    } catch {
      setTestResult({ success: false, message: '分析失败' })
    } finally {
      setAnalyzing(false)
    }
  }

  const scoreColor = (score: number) => {
    if (score >= 0.3) return 'text-green-400'
    if (score <= -0.3) return 'text-red-400'
    return 'text-gray-400'
  }

  const scoreBar = (score: number) => {
    const pct = ((score + 1) / 2) * 100
    return (
      <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${score >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: `${pct}%`, marginLeft: score < 0 ? `${pct}%` : '50%', maxWidth: '50%' }}
        />
      </div>
    )
  }

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">AI 模型设置</h1>
        <p className="text-sm text-gray-400 mt-1">配置 LLM 模型用于行业情绪分析，辅助轮动信号决策</p>
      </div>

      {/* Provider Selection */}
      <div className="space-y-3">
        <label className="block text-xs text-gray-400">选择模型提供商</label>
        <div className="grid grid-cols-3 gap-3">
          {providers.map(p => (
            <button
              key={p.id}
              type="button"
              onClick={() => handleProviderChange(p.id)}
              className={`p-3 rounded-lg border text-left transition-all ${
                provider === p.id
                  ? 'border-blue-500 bg-blue-900/20'
                  : 'border-gray-700 bg-gray-800 hover:border-gray-600'
              }`}
            >
              <div className="text-sm font-medium text-white">{p.name}</div>
              <div className="text-xs text-gray-500 mt-1">
                {p.models.length} 个模型
                {p.supports_web_search && <span className="ml-1 text-green-500">• 联网</span>}
              </div>
            </button>
          ))}
          {/* Custom */}
          <button
            type="button"
            onClick={() => handleProviderChange('custom')}
            className={`p-3 rounded-lg border text-left transition-all ${
              provider === 'custom'
                ? 'border-blue-500 bg-blue-900/20'
                : 'border-gray-700 bg-gray-800 hover:border-gray-600'
            }`}
          >
            <div className="text-sm font-medium text-white">自定义</div>
            <div className="text-xs text-gray-500 mt-1">OpenAI 协议兼容</div>
          </button>
        </div>
      </div>

      {/* Config Form */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-4">
        {/* API Base (custom or override) */}
        {isCustom ? (
          <div>
            <label className="block text-xs text-gray-400 mb-1">API Base URL</label>
            <input
              type="url"
              value={apiBase}
              onChange={e => setApiBase(e.target.value)}
              placeholder="https://your-provider.com/v1"
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">需兼容 OpenAI /chat/completions 协议</p>
          </div>
        ) : (
          <div>
            <label className="block text-xs text-gray-400 mb-1">API 地址</label>
            <div className="text-sm text-gray-300 bg-gray-700/50 px-3 py-2 rounded-md font-mono text-xs">
              {currentProvider?.api_base || '—'}
            </div>
          </div>
        )}

        {/* API Key */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder={config?.has_api_key ? '已保存（留空则不修改）' : 'sk-xxxxxxxx'}
            className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Model */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">模型</label>
          {isCustom ? (
            <input
              type="text"
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              placeholder="gpt-4o / claude-sonnet-4-6 / ..."
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          ) : (
            <select
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {availableModels.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name} {m.supports_search ? '(支持联网)' : ''}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Web Search Toggle */}
        {supportsSearch && (
          <div className="flex items-center justify-between py-2">
            <div>
              <label className="text-sm text-white">启用联网检索</label>
              <p className="text-xs text-gray-500">分析时搜索最新财经新闻和政策动态</p>
            </div>
            <input
              type="checkbox"
              checked={webSearch}
              onChange={e => setWebSearch(e.target.checked)}
              className="w-5 h-5 accent-blue-500"
            />
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3 pt-2">
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {saving ? '保存中...' : '保存配置'}
          </button>
          <button
            onClick={handleTest}
            disabled={testing || !config?.has_api_key}
            className="px-4 py-2.5 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm rounded-lg transition-colors"
          >
            {testing ? '测试中...' : '测试连接'}
          </button>
        </div>

        {/* Test Result */}
        {testResult && (
          <div className={`text-sm px-3 py-2 rounded-lg ${
            testResult.success ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
          }`}>
            {testResult.message}
          </div>
        )}
      </div>

      {/* Sentiment Analysis Section */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">情绪分析</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              对全池 ETF 分析行业情绪，作为轮动信号的背离过滤器
            </p>
          </div>
          <button
            onClick={handleAnalyze}
            disabled={analyzing || !config?.has_api_key}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white text-xs font-medium rounded-lg transition-colors"
          >
            {analyzing ? '分析中...' : '触发分析'}
          </button>
        </div>

        {/* Filter logic explanation */}
        <div className="text-xs text-gray-500 bg-gray-900/50 rounded-lg p-3 space-y-1">
          <p className="text-gray-400 font-medium">V1 过滤器规则：</p>
          <p>• 情绪 ≥ -0.3：不干预动量选股</p>
          <p>• 情绪 &lt; -0.3 且在 Top N 中：<span className="text-yellow-400">警告 + 降权 50%</span></p>
          <p>• 情绪 &lt; -0.6 且在 Top N 中：<span className="text-red-400">剔除，候补递补</span></p>
        </div>

        {/* Results */}
        {sentiments.length > 0 && (
          <div className="space-y-1 max-h-80 overflow-y-auto">
            {sentiments.map(item => (
              <div
                key={item.code}
                className="flex items-center gap-3 px-3 py-2 rounded-lg bg-gray-900/40 hover:bg-gray-900/60 transition-colors group"
              >
                <span className={`text-sm font-bold w-12 text-right ${scoreColor(item.score)}`}>
                  {item.score > 0 ? '+' : ''}{item.score.toFixed(2)}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 font-mono">{item.code}</span>
                    {scoreBar(item.score)}
                  </div>
                  <p className="text-xs text-gray-500 truncate group-hover:whitespace-normal mt-0.5">
                    {item.raw_text?.replace(/SCORE:.*/, '').trim().slice(0, 100)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

        {sentiments.length === 0 && config?.has_api_key && (
          <p className="text-xs text-gray-500 text-center py-4">暂无分析数据，点击"触发分析"开始</p>
        )}
        {!config?.has_api_key && (
          <p className="text-xs text-gray-500 text-center py-4">请先保存 LLM 配置</p>
        )}
      </div>
    </div>
  )
}

export default LLMSettingsPage
