# Share Monitor

## 美股科技市场温度计 MVP

仓库新增了一个可独立运行的美股日频 MVP，用于回答：

- 当前美国科技市场是 `risk_on`、`neutral` 还是 `risk_off`；
- 宏观、波动与趋势分别贡献了多少风险；
- QQQ 总风险预算应该积极、中性还是防守；
- AAPL、MSFT、NVDA 等自选股当前处于允许建仓、小仓、持有、观察减仓或退出条件触发状态。
- MU–SOXS 配对策略当前的买入 MU 与卖出 MU / 买入 SOXS 置信度，以及板块确认和冲击冷却状态。

它是风险环境和交易许可监测器，不预测精确顶底，不连接券商，也不自动下单。

### 快速运行

```powershell
python -m pip install -r requirements-us.txt
python run_us_monitor.py build
python run_us_monitor.py doctor
python run_us_monitor.py serve
```

打开 <http://127.0.0.1:8788/web/>。Windows 也可直接运行：

```powershell
.\scripts\windows\start_us_tech.ps1
```

运行测试：

```powershell
python -m pytest
```

### 在线访问与数据刷新

公开看板：<https://slimold.github.io/share-monitor/>

- Windows、macOS、Android 与 iPhone 浏览器都可以直接访问，页面按窄屏自动改为单列布局。
- GitHub Actions 在北京时间周二至周六 09:17 自动运行，对应前一个美股交易日收盘后；每次都会重新抓取行情、运行测试和回测，再发布最新快照。
- 页面右上角“重新读取”用于绕过浏览器缓存并读取最近一次已发布数据；它不会直接触发上游行情抓取。
- 如需立即抓取并发布，可打开仓库的 **Actions → Deploy US Tech Monitor → Run workflow** 手动运行。

部署工作流见 [deploy-us-tech-monitor.yml](.github/workflows/deploy-us-tech-monitor.yml)。发布物只包含静态页面和汇总 JSON，不包含 `.env`、原始行情缓存或任何密钥。

构建命令会产生：

- `data/us_tech_snapshot.json`：当前风险、数据质量、历史温度和自选股状态；
- `data/backtest_results.json`：五年正式评估区间、末尾 252 日评估切片、逐日仓位/成本/收益、净值曲线和敏感性实验；
- `reports/backtest_latest.md`：可阅读的回测报告。

模型评估只统计最近五年；评估期之前的 450 个日历日只用于 200 日均线和历史分位预热，不计入绩效。信号在 t 日收盘后生成，于下一交易日收盘执行，从再下一段收盘到收盘收益开始生效；策略与 QQQ 基准都计入 10 bp 初始建仓成本，策略调仓另按同一费率计费。

当前可复现示例（市场数据截至 2026-09-25）：

| 区间 | 策略 CAGR | QQQ CAGR | 策略最大回撤 | QQQ 最大回撤 | 策略 Sharpe | QQQ Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| 近五年 | 12.42% | 16.42% | -12.77% | -35.12% | 0.82 | 0.61 |
| 末尾 252 日评估切片 | 10.80% | 25.36% | -5.49% | -11.96% | 0.74 | 1.04 |

同期最新温度计为 **47.73 / 100（neutral）**，候选 QQQ 风险仓位为 **55%**：宏观风险偏高、波动风险较低、趋势风险居中。它表达的是风险预算环境，不是“明天涨跌”的概率。

这个结果表明默认保守策略明显压低了回撤和波动，但预声明的验证门槛只通过了“回撤更浅”一项：CAGR 只保留同期 QQQ 的 42.6%，Sharpe 也低于 QQQ。简单 QQQ 200DMA 基线在该切片的 CAGR 为 20.77%、Sharpe 为 0.89，均高于默认复合策略，代价是 -11.22% 的更深回撤。因此它已经是一个可运行的风控 MVP，但不能声称已经是经过验证的自动买卖或超额收益策略。

详细设计见 [MVP 开发计划](docs/US_TECH_MVP_PLAN.md)、[方法与回测实验](docs/US_TECH_METHODOLOGY.md) 和 [MU–SOXS 个股置信度策略 v2.1](docs/MU_SOXS_CONFIDENCE_V2.md)。

---

## 旧版：A股风险监视器 · A-RISK/MONITOR

一个本地运行的 A 股大盘风险监测看板：ERP 股权风险溢价、万得全A PE、10Y 国债、破净率、两市成交额×换手率、HV30 波动率、信贷脉冲（社融存量同比一阶导）、两融余额+动量、ETF 资金流向、申万行业热力图，以及一个「两层漏斗决策模型」给出综合仓位建议。

数据每交易日收盘后自动抓取（AKShare + 央行官网直连 + 新浪/东财），本地静态页面渲染，**无需任何后端服务器**。

![看板首页](docs/screenshot.png)

> 📊 [查看完整长图（含 ETF 资金流向、行业热力图、决策模型）](docs/screenshot-full.png)

> ⚠️ 本项目仅供研究学习，所有指标不构成投资建议。投资有风险，入市需谨慎。

## 组成

| 文件 | 作用 |
|---|---|
| `arisk_monitor_local.html` | 看板本体（Chart.js 走 CDN，其余内联） |
| `update_arisk_data.py` | 抓全量数据 → 生成 `arisk_data.json`（约 95 秒） |
| `proxy.py` | 本地代理(8899)，供浏览器盘中实时抓数 + 妙想API 转发 |
| `check_and_update.sh` | 判断数据是否落后于最新交易日，落后才更新 |
| `run_arisk_update.sh` | 跑一次更新（被 check 调用，或手动） |
| `start.sh` / `stop.sh` | 一键起停（代理 8899 + 静态服务器 8788） |
| `arisk_data.json` | 数据快照（仓库内为种子数据，跑一次更新即刷新） |

## 快速开始

```bash
git clone <你的仓库地址> arisk
cd arisk

# 1. 建虚拟环境 + 装依赖（需 Python 3.9+）
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2.（可选）配妙想 API key；不配也能跑，社融走央行直连
cp .env.example .env
#   然后编辑 .env 填入 MX_APIKEY

# 3. 首次抓数
./venv/bin/python update_arisk_data.py

# 4. 启动并打开看板
bash start.sh
```

看板地址：<http://localhost:8788/arisk_monitor_local.html>

> ⚠️ **必须通过 `start.sh`（本地 http）打开，不能直接双击 HTML**——`file://` 协议下浏览器禁止读取本地 JSON，页面会空白。

停止服务：`bash stop.sh`

## 每日自动更新

数据只在**交易日收盘后（约 18:00 起）**发布，`check_and_update.sh` 会判断当前数据是否已覆盖最新交易日：已覆盖则秒退，落后才抓。

- **macOS（launchd）**：见 `com.arisk.update.plist.example`，把 `__ARISK_DIR__` 换成本目录绝对路径后装入 `~/Library/LaunchAgents/`，每天 16:10–22:10 每小时判断一次。
- **Linux（cron）**：`crontab -e` 添加
  ```
  10 16-22 * * * /bin/bash /path/to/arisk/check_and_update.sh
  ```

手动立即更新：`bash run_arisk_update.sh`

## 已知限制

- **数据源在中国境内**（东财/新浪/央行）。海外服务器直连可能受限或较慢，`proxy.py` 已用 `curl_cffi` 模拟 Chrome TLS 指纹绕过部分反爬；仍不通时需自行加代理。
- 本仓库的 `proxy.py` 为**通用反代版**，未实现看板期望的部分语义端点（`/pe` `/bond` `/index_kline` 等）。这些盘中 live-refresh 会回退到 `arisk_data.json`——由于该 JSON 每日自动更新，数据整体仍是新的，只是缺分钟级盘中刷新。
- 妙想 API（`MX_APIKEY`）为可选增强，缺省走 AKShare/央行回退。

## 安全

`.env` 含你的 API key，已被 `.gitignore` 排除。**切勿把真实 `.env` 提交或分享。** 如误提交，请立即在东财后台吊销并更换 key。
