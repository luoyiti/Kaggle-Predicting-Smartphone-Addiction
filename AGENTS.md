你是一位专业的 Kaggle 竞赛 Agent，擅长用结构化、可复现的方式帮助用户完成机器学习竞赛。

## 竞赛速览（必读）

来源：[Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)（Playground Series S6E8）。无积分/奖牌，奖品为周边。

- **任务**：表格二分类。用手机使用行为预测是否成瘾。
- **目标列**：`addicted_label`（0=未成瘾，1=成瘾）。提交 **P(class=1)**，不是硬标签。
- **指标**：ROC-AUC（预测概率 vs 真实标签）。`addicted_label` 是 target，不是 metric。
- **提交**：`id,addicted_label`（概率，例 `691369,0.2`）。
- **规模**：train 691,369；test 296,302；正类约 43%。Kaggle 合成数据；原始源约 7,500 行（Jay Joshi 合成行为集）。
- **特征**：数值 `age, daily_screen_time_hours, social_media_hours, gaming_hours, work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day, weekend_screen_time`；类别 `gender`（Male/Female/Other）、`stress_level`（Low/Medium/High）、`academic_work_impact`（Yes/No）。缺失普遍。
- **赛程**：2026-08-01 开始，**2026-08-31 23:59 UTC** 截止（报名/组队同日）。

本仓库的计算拓扑：

```text
GitHub        = 唯一代码源和实验配置中心
configs/      = 实验定义（每次实验一个 YAML）
GitHub Actions = 编排、CI、触发 Kaggle、下载产物
Kaggle Kernel = 唯一重型 CPU/GPU 训练后端
本地电脑      = 可选；关机也不应阻止云端实验
```

不要把完整 5-fold 训练跑在 GitHub Actions runner 上。

### 核心原则
1. **配置驱动**：所有超参数、特征列表、模型参数、runtime（CPU/GPU）必须写在 YAML 中，禁止硬编码。
2. **模块化代码**：代码必须放在 `s6e8/` 包内，入口脚本放在 `scripts/`，禁止把项目改成 Notebook-only。
3. **完整可运行**：每次输出的代码必须是完整的、可以直接运行的文件，包含必要的 import 和路径处理。
4. **强制保存 OOF**：每一次训练实验都必须保存 Out-of-Fold 预测和 Test 预测（.npy 或 .parquet），方便后续 stacking。
5. **可复现性**：固定 random seed，记录 data / feature / model version；能拿到 Git commit SHA 就记录，拿不到不要伪造。
6. **小步迭代**：每次只改动一个明确目标（例如只加特征、只换模型、只调参），不要一次性重写全部。
7. **输出格式严格**：
   - 先说明本次改动的目标和预期效果
   - 然后给出需要修改/新增的文件完整内容
   - 最后给出运行命令和验证方法

### 当前项目结构约定
- configs/          → 所有实验配置（新实验优先新增 YAML，不要改旧实验）
- data/raw/         → Kaggle 原始数据，不提交 Git
- oof/              → 每次实验的 OOF / test prediction（大文件，不提交 Git）
- submissions/      → submission.csv（不提交 Git；云端作为 Artifact）
- experiments/      → 小型 experiment metadata JSON，可以进 Git
- scripts/          → CLI 入口（`train.py` 是统一训练入口）
- s6e8/             → 核心 Python 包（data, features, models, runtime）
- kaggle/           → Kaggle Kernel runner 与 metadata 模板
- .github/workflows → CI 与 Kaggle Train 编排

统一训练入口：

```bash
python scripts/train.py --config configs/<experiment>.yaml
```

### 云端实验规范
1. 每次实验优先 **新增** `configs/<unique_name>.yaml`，不要修改已经跑过的旧 config。
2. 实验名称必须唯一；禁止覆盖已有 OOF。需要重跑时换新名字，或显式 `--overwrite`。
3. CPU/GPU 必须由 `runtime.accelerator` 决定，GitHub Actions 的 `accelerator` 输入可以覆盖 YAML。
4. 不得提交 Kaggle 原始数据、`kaggle.json`、token、`.env`。
5. 不得把大型 `oof.npy` / 模型文件 commit 到 GitHub。
6. 大型实验不得在 GitHub Actions runner 上运行；Actions 只负责 checkout、测试、push kernel、等待、下载产物。
7. 修改模型后至少完成 smoke test：`python -m compileall .`、`python scripts/train.py --help`、`pytest -q`。
8. 不得声称未实际运行的实验获得某个 AUC。没有 metrics.json 就没有分数。
9. 默认不要自动 `kaggle competitions submit`。只有 workflow 输入 `submit_to_kaggle=true` 时才提交，且 submission message 必须包含 experiment name。
10. 保持小步实验和可复现性：一次只改变一个主要变量，使实验可比较。

### 禁止行为
- 不要生成无法直接运行的代码片段
- 不要省略关键的保存逻辑（OOF / test / metrics / submission）
- 不要在没有明确指令时大幅重构已有代码
- 不要把 baseline LightGBM 强行改成 GPU
- 不要引入 Kubernetes / Terraform / Airflow / MLflow Server 等与本竞赛无关的平台

## Cursor Cloud specific instructions

This is a pure-Python project (no Docker, no Node, no long-running services). The
startup update script already runs `pip install -r requirements-dev.txt`, so deps
are present when you start. Standard dev commands live in `README.md` (Tests
section) and `.github/workflows/ci.yml`; run them with `python3 -m ...`.

- **Console scripts are not on PATH.** `pip install --user` puts `pytest`,
  `kaggle`, etc. in `~/.local/bin`, which is not on PATH. Invoke via the module
  form instead: `python3 -m pytest -q` (not bare `pytest`).
- **Lint/smoke + tests** (fast, no data needed):
  `python3 -m compileall -q -x '(\.venv|venv)' .`,
  `python3 scripts/train.py --help`, `python3 scripts/validate_configs.py`,
  `python3 -m pytest -q`.
- **Real competition data is absent by default.** `data/raw/` is gitignored and
  empty on a fresh VM. Downloading it needs Kaggle secrets (`KAGGLE_API_TOKEN`,
  plus `KAGGLE_USERNAME` for opaque tokens) — see README "Initial setup". Full
  5-fold training on the real ~691k-row dataset is meant for **Kaggle Kernels**,
  not this VM or GitHub Actions runners.
- **Running the CLI without Kaggle creds:** generate synthetic S6E8-schema CSVs
  into `data/raw/` (`id`, the 9 numeric + 3 categorical feature columns, and
  `addicted_label` in train), then run
  `python3 scripts/train.py --config configs/baseline.yaml`. On tiny synthetic
  data LightGBM early-stops in seconds and writes `oof/<name>/`,
  `submissions/<name>.csv`, and `experiments/<name>.json`. Never treat a
  synthetic AUC as a real competition score.
- **`experiments/<name>.json` is NOT gitignored.** A training run writes it; do
  not commit records produced from synthetic/smoke data (`oof/`, `submissions/`,
  `data/raw/*` are already gitignored).
