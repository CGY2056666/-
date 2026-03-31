# 专注小窝 V1

一个基于 Flask 的本地网页工具，用来记录一整段专注过程，并结合截图内容判断当前画面是否仍然服务于你的目标。

当前版本支持两条主流程：

- 实时专注记录：输入目标和时长后共享屏幕，系统会持续截图、分析并生成整段报告
- 手动上传截图：补充上传历史截图，快速查看这一批画面的专注度结果

## 当前能力

- 目标关键词提取与场景建模
- 通过 SiliconFlow 多模态模型做截图识别和专注评分
- 本地时长推荐模型
  - 不再调用 DeepSeek API
  - 会结合目标类型、任务复杂度、历史记录、完成率和注意力衰减趋势给出推荐
- 实时会话管理
  - 自动累计截图
  - 支持分心提醒
  - 结束后自动写入 `focus_history.json`

## 运行环境

- Python 3.11+
- Windows 本地运行已验证
- 依赖见 `requirements.txt`

安装依赖：

```bash
pip install -r requirements.txt
```

## 启动方式

```bash
python app.py
```

或直接运行：

```bash
run_focus_site.bat
```

启动后访问：

`http://127.0.0.1:5000`

## 配置项

项目会自动读取根目录下的 `.env`。

至少需要配置：

```bash
SILICONFLOW_API_KEY=your_api_key
```

常用可选项：

```bash
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=Qwen/Qwen3-VL-8B-Instruct
SILICONFLOW_TIMEOUT_SECONDS=
FOCUS_FRAME_REQUEST_TIMEOUT_MS=
FOCUS_SESSION_DEFAULT_DURATION_MINUTES=25
FOCUS_SESSION_MIN_DURATION_MINUTES=1
FOCUS_SESSION_MAX_DURATION_MINUTES=180
FOCUS_APP_SECRET=focus-app-dev-secret
```

说明：

- `SILICONFLOW_TIMEOUT_SECONDS` 留空时，后端请求不额外设置超时上限
- `FOCUS_FRAME_REQUEST_TIMEOUT_MS` 留空时，前端实时截图上传不主动中断请求
- `focus_history.json` 会保存历史专注记录和下一次推荐结果

## 目录结构

```text
app.py
focus_history.json
run_focus_site.bat
requirements.txt
templates/
  index.html
static/
  app.js
  styles.css
focus_engine/
  __init__.py
  config.py
  duration_recommender.py
  goal_profiles.py
  history_storage.py
  models.py
  ocr.py
  pipeline.py
  scoring.py
  session.py
  siliconflow_vlm.py
  utils.py
```

## 核心模块说明

- `app.py`
  Flask 入口，负责页面路由和 API。
- `focus_engine/pipeline.py`
  上传分析和单帧分析总入口。
- `focus_engine/siliconflow_vlm.py`
  SiliconFlow 多模态调用与结果归一化。
- `focus_engine/session.py`
  实时会话管理、分心提醒、历史落盘。
- `focus_engine/duration_recommender.py`
  本地时长推荐模型。
- `focus_engine/scoring.py`
  多帧平滑、状态判定、汇总统计。

## 主要接口

- `GET /`
  首页。
- `POST /analyze`
  手动上传截图并渲染网页报告。
- `POST /api/analyze`
  手动上传截图并返回 JSON。
- `POST /api/duration/recommend`
  根据目标返回本地推荐时长。
- `POST /api/session/start`
  开启实时会话。
- `POST /api/session/<session_id>/frame`
  追加一张实时截图。
- `GET /api/session/<session_id>`
  查看当前会话状态。
- `POST /api/session/<session_id>/complete`
  主动结束会话并返回最终结果。
- `GET /health`
  查看服务健康状态和运行时配置摘要。

## 输出数据

- 手动上传会返回：
  - 目标画像
  - 每张截图的专注状态与原因
  - 整体汇总指标和建议
- 实时模式会额外返回：
  - 当前会话进度
  - 分心提醒信号
  - 当前轮推荐时长
  - 下一轮推荐时长
  - AI 复盘寄语
