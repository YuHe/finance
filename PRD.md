# PRD：A股行业ETF轮动量化系统 v2

## 一、产品定位

面向个人量化投资者的 **A股行业ETF周频轮动系统**，包含回测验证、信号推荐、模拟盘跟踪三大模块，**严格对齐实盘交易规则**，确保模拟结果可直接指导实盘操作。

---

## 二、核心原则

1. **模拟盘=实盘** — T+1延迟执行、涨跌停检查、成交量约束、最小交易单位，所有约束在回测和模拟盘中统一建模
2. **简单有效** — 5层架构代替9层，周频调仓，持仓3-5只，避免过度优化
3. **人机协作** — 系统给出信号，用户决定是否采纳，系统记录并追踪收益
4. **可视化驱动** — K线+买卖点、收益曲线、多维度评价指标一目了然

---

## 三、系统架构（5层）

```
数据层(Data) → 因子层(Factor) → 环境层(Regime) → 选股层(Selection) → 执行层(Execution)
```

### 整体技术栈

| 分类 | 选型 |
|------|------|
| 前端 | React 18 + TypeScript + Vite + Tailwind + ECharts |
| 后端 | FastAPI + SQLAlchemy + PostgreSQL |
| 数据源 | akshare（东方财富） |
| 计算 | numpy / pandas |
| 部署 | Docker Compose |

---

## 四、各层功能需求

### 4.1 数据层 (`data_layer`)

| 功能 | 说明 |
|------|------|
| 历史日频数据 | OHLCV + 换手率 + 成交额，后复权，增量更新 |
| 交易日历 | A股交易日历，节假日/停牌标记 |
| 涨跌停标记 | 每日计算涨跌停状态（涨停=涨幅≥9.8%，跌停=跌幅≤-9.8%） |
| 成交额数据 | 用于计算可建仓容量 |
| ETF池管理 | 可配置的行业ETF池（代码+名称+行业分类） |
| 存储 | PostgreSQL（支持前端并发读写） |

**ETF池（默认）**：
- 半导体(512480)、银行(512800)、医药(512010)、新能源(516160)
- 信息技术(512330)、券商(512000)、军工(512660)、消费(159928)
- 有色金属(512400)、煤炭(515220)、房地产(512200)、食品饮料(515170)
- 基建(516950)、汽车(516110)、通信(515880)

### 4.2 因子层 (`factor_layer`)

精简为三类核心因子：

| 因子 | 计算方式 | 用途 |
|------|----------|------|
| 动量 | 20日涨幅（收盘价/20日前收盘价 - 1） | 趋势强度排名 |
| 趋势 | MA5 > MA20 且 MA20 > MA60 | 趋势方向过滤 |
| 量价确认 | 5日均成交额 > 20日均成交额 | 资金流入确认 |

**因子综合评分**：
- 动量排名分（0-1，cross-sectional rank）
- 趋势加分（符合+0.2，不符合+0）
- 量价加分（放量+0.1，缩量+0）

### 4.3 环境层 (`regime_layer`)

简化为单一维度判断：

| 环境 | 判定条件 | 现金比例 |
|------|----------|----------|
| 牛市 | 沪深300指数 > MA60 且 MA20斜率>0 | 0% |
| 震荡 | 沪深300指数 > MA60 但 MA20斜率≤0 | 30% |
| 熊市 | 沪深300指数 < MA60 | 60% |
| 极端熊市 | 沪深300指数 < MA60 且 20日跌幅>10% | 100% |

基准标的：沪深300ETF（510300）

### 4.4 选股层 (`selection_layer`)

| 功能 | 说明 |
|------|------|
| 排名 | 按因子综合评分降序 |
| 过滤 | 剔除趋势不符合的（MA5<MA20）|
| 选取 | 取Top N只（默认N=3，可配置1-5） |
| 加权 | 等权 或 逆波动率加权（可配置） |
| 现金分配 | 根据环境层现金比例，剩余资金分配给持仓ETF |

### 4.5 执行层 (`execution_layer`)

**核心：确保模拟盘/实盘零差异**

| 约束 | 规则 |
|------|------|
| T+1延迟 | 周五收盘计算信号 → 下周一开盘价执行（或指定日计算→次日执行） |
| 涨停不可买 | 目标ETF若次日涨停（开盘即涨停），跳过买入，保留现金 |
| 跌停不可卖 | 持仓ETF若次日跌停（开盘即跌停），延迟卖出 |
| 成交量约束 | 单只ETF单日买入金额 ≤ 前5日日均成交额的5% |
| 最小交易单位 | ETF最小100份，按开盘价计算可买份数 |
| 交易成本 | 买入0.05%（佣金）+ 卖出0.05%（佣金），无印花税 |
| 价格假设 | 回测/模拟盘统一用次日开盘价（最贴近实盘可执行价） |

---

## 五、调仓频率

| 选项 | 说明 |
|------|------|
| 周频（默认） | 每周五收盘后计算，下周一开盘执行 |
| 双周频 | 每两周调一次 |
| 月频 | 每月最后一个交易日计算 |
| 自定义 | 前端可选调仓日 |

---

## 六、风控规则

| 层级 | 规则 | 动作 |
|------|------|------|
| 个股止损 | 单只ETF持仓亏损 > 8% | 下一调仓日强制卖出 |
| 组合止损 | 组合净值回撤 > 12% | 全部减仓至50% |
| 熔断 | 组合净值回撤 > 20% | 全部清仓，暂停信号2周 |

---

## 七、前端功能

### 7.1 回测模块

| 功能 | 说明 |
|------|------|
| 参数配置面板 | 时间范围、调仓频率、持仓数N、加权方式、初始资金 |
| 策略参数 | 动量窗口、MA参数、止损线、环境判断参数 |
| K线图 | ECharts K线 + 买入卖出标记点（箭头）|
| 收益曲线 | 策略净值 vs 沪深300基准，双Y轴 |
| 持仓时间线 | 每周持仓变化甘特图 |
| 指标面板 | 年化收益、年化波动、Sharpe、Sortino、最大回撤、胜率、盈亏比、换手率 |
| 分年度/月度表格 | 每月/每年收益热力图 |
| 回撤图 | 回撤曲线（水下曲线） |

### 7.2 信号模块（每日/每周）

| 功能 | 说明 |
|------|------|
| 当前持仓 | 展示当前模拟盘持仓（如有） |
| 本周信号 | 推荐买入/卖出标的及目标权重 |
| 因子明细 | 每只ETF的动量/趋势/量价得分 |
| 环境状态 | 当前市场环境（牛/震荡/熊）及现金比例建议 |
| 采纳按钮 | 用户点击"采纳"后执行，"忽略"则跳过 |

### 7.3 模拟盘模块

| 功能 | 说明 |
|------|------|
| 持仓管理 | 当前持有ETF、成本价、市值、盈亏 |
| 交易记录 | 历史所有买卖记录（时间、标的、方向、价格、数量、费用） |
| 收益追踪 | 累计收益曲线、每日盈亏柱状图 |
| 基准对比 | vs 沪深300 同期表现 |
| 偏差监控 | 模拟盘信号 vs 实际可执行性（涨跌停/流动性不足时标红告警） |

---

## 八、用户系统与鉴权

### 8.1 设计原则

- 面向**私有部署**场景，无公开注册入口（注册需知道系统地址）
- **首位注册用户自动成为管理员**，无需预置账号或运行初始化脚本
- 使用 **JWT Bearer Token**（7天有效期），存于 `localStorage`；无 OAuth/SSO 复杂依赖
- 鉴权以 FastAPI `Depends` 装饰在路由注册层，各业务 router 保持无感知

### 8.2 角色与权限

| 角色 | 权限 |
|------|------|
| 普通用户 | 访问全部业务功能（回测/信号/模拟盘/市场数据） |
| 管理员 | 普通用户权限 + 用户管理（启用/禁用/设权限/重置密码/删除） |

### 8.3 认证 API（无需登录可访问）

```
POST  /api/v1/auth/register          — 注册（用户名/邮箱/密码，首位注册者自动为管理员）
POST  /api/v1/auth/login             — 登录（返回 JWT access_token + 用户信息）
GET   /api/v1/auth/me                — 获取当前用户信息（需登录）
```

### 8.4 管理员 API

```
GET    /api/v1/auth/admin/users          — 用户列表
PATCH  /api/v1/auth/admin/users/{id}     — 更新用户（is_active / is_admin / password）
DELETE /api/v1/auth/admin/users/{id}     — 删除用户
```

### 8.5 前端路由保护

| 路由 | 保护 |
|------|------|
| `/login` | 公开 |
| `/backtest` `/signal` `/portfolio` | 需登录（`RequireAuth`） |
| `/admin` | 需登录且为管理员（`RequireAdmin`） |

### 8.6 部署说明（域名 finance.iwill.cc）

docker-compose 本身不需要感知域名，域名路由在宿主机反向代理层处理：

```
用户 → finance.iwill.cc:443 → 宿主机 Nginx/Caddy → localhost:8193 → 容器
```

docker-compose 唯一需要配置的是 `SECRET_KEY`（通过 `.env` 文件设置），CORS 白名单已包含 `https://finance.iwill.cc`。

---

## 九、API设计

### 后端路由

```
POST   /api/v1/backtest/run          — 运行回测（参数化）
GET    /api/v1/backtest/result/{id}   — 获取回测结果

GET    /api/v1/signal/latest          — 获取最新信号
GET    /api/v1/signal/history         — 历史信号列表
POST   /api/v1/signal/adopt/{id}      — 采纳信号

GET    /api/v1/portfolio/positions    — 当前持仓
GET    /api/v1/portfolio/trades       — 交易记录
GET    /api/v1/portfolio/performance  — 收益表现

GET    /api/v1/market/etfs            — ETF池列表
GET    /api/v1/market/kline/{code}    — K线数据
GET    /api/v1/market/regime          — 当前市场环境

PUT    /api/v1/settings/strategy      — 更新策略参数
GET    /api/v1/settings/strategy      — 获取策略参数
```

---

## 十、数据模型（核心表）

```sql
-- 用户（SQLite，独立 users.db）
users (id, username, email, hashed_password, is_admin, is_active, created_at, last_login)

-- ETF池
etf_pool (code, name, industry, active, created_at)

-- 日频行情
etf_daily (code, date, open, high, low, close, volume, amount, turnover, is_limit_up, is_limit_down)

-- 回测结果
backtest_result (id, params_json, result_json, created_at)

-- 信号
signal (id, date, signal_json, regime, status[pending/adopted/ignored], created_at)

-- 模拟盘持仓
portfolio_position (id, code, shares, cost_price, current_price, pnl, updated_at)

-- 交易记录
portfolio_trade (id, signal_id, code, direction[buy/sell], price, shares, fee, executed_at)

-- 每日净值
portfolio_nav (date, nav, benchmark_nav, cash, total_value)
```

---

## 十一、核心设计理念

> **简单可执行 > 复杂不可验。**
> 周频轮动+趋势过滤+环境感知+严格实盘约束 = 可信赖的系统。

---

## 十二、与旧版差异

| 维度 | v1（旧） | v2（新） |
|------|----------|----------|
| 层数 | 9层 | 5层 |
| 频率 | 日频 | 周频 |
| 优化器 | CVaR+cvxpy | 等权/逆波动率 |
| 对冲 | Beta对冲 | 现金比例调节 |
| T+1 | 未建模 | 严格建模 |
| 涨跌停 | 未处理 | 每日检查 |
| 成交量 | 未约束 | 5%上限 |
| 前端 | 无 | React+ECharts |
| 模拟盘 | 无 | 完整支持 |
| 数据库 | SQLite | PostgreSQL + SQLite(用户) |
| 鉴权 | 无 | JWT + 管理员页面 |
