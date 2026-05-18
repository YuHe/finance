# A股行业ETF轮动量化系统

## 快速开始

### 1. 安装依赖

```bash
# 后端
cd backend && pip install -r requirements.txt

# 前端
cd frontend && npm install
```

### 2. 更新数据

```python
from data_layer import DataManager
dm = DataManager()
dm.update_all()  # 首次运行需下载历史数据
```

### 3. 运行回测

```python
from backtest import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date="2019-01-01",
    end_date="2024-12-31",
    top_n=3,
    weight_method="equal",
)
engine = BacktestEngine(config)
result = engine.run()
```

### 4. 启动前后端

```bash
# 终端1 - 后端
cd backend && uvicorn main:app --reload --port 8000

# 终端2 - 前端
cd frontend && npm run dev
```

访问 http://localhost:5173

## 架构

```
数据层(Data) → 因子层(Factor) → 环境层(Regime) → 选股层(Selection) → 执行层(Execution)
```

详见 [PRD.md](./PRD.md)
