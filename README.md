# Focus Compass

这是一个基于 Flask 的本地网页工具，用来分析一组学习截图是否持续服务于当前专注目标。

当前版本的实际能力是：

- 用户自由输入本次专注目标
- 批量上传截图并生成分析报告
- 支持 OCR、目标词抽取、DeepSeek 评分和时间窗修正
- 提供基础的 Session API，便于后续接实时截图流

## 当前项目结构

```text
app.py
focus_engine/
  __init__.py
  config.py
  deepseek_scoring.py
  goal_profiles.py
  models.py
  ocr.py
  pipeline.py
  scoring.py
  session.py
  utils.py
templates/
  index.html
static/
  styles.css
requirements.txt
run_focus_site.bat
测试.png
```

## 主要文件

- `app.py`
  Flask 入口，提供页面路由、批量分析接口和 Session API。
- `focus_engine/`
  核心分析逻辑，包括 OCR、目标理解、评分、汇总和实时会话管理。
- `templates/index.html`
  页面模板。
- `static/styles.css`
  页面样式。
- `测试.png`
  演示分析接口使用的本地样例图。

## 启动方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置 Tesseract OCR

默认会自动尝试常见 Windows 安装路径；如果你的安装路径不同，可以设置环境变量：

```bash
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

3. 如需启用 DeepSeek 评分，请在本地 `.env` 或系统环境变量中配置：

```bash
set DEEPSEEK_API_KEY=your_api_key
```

4. 启动项目

```bash
python app.py
```

也可以直接运行：

```bash
run_focus_site.bat
```

5. 打开浏览器访问：

`http://127.0.0.1:5000`

## 主要接口

- `POST /analyze`
  表单上传截图并渲染网页报告。
- `POST /api/analyze`
  返回 JSON 结果。
- `POST /api/session/start`
  开启一个实时分析会话。
- `POST /api/session/<session_id>/frame`
  追加一张截图到会话窗口。
- `GET /api/session/<session_id>`
  查看当前会话状态。
- `GET /api/analyze-demo`
  使用 `测试.png` 走一遍演示分析。

## 清理说明

项目已经去掉历史实验档案、重复网页文件、缓存目录和无运行时作用的规划文档。当前目录只保留运行这个网页应用所需的核心文件。
