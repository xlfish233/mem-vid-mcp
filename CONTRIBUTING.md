# Contributing

感谢你愿意为 `memvid-mcp` 贡献代码。

## 开发环境

- Python `>=3.10`
- 建议使用 `uv`（也可用 `pip`/`venv`）

安装依赖（开发用）：

```bash
pip install -e ".[dev]"
```

## 质量门禁

运行测试：

```bash
pytest -q
```

运行静态检查：

```bash
ruff check src tests
```

## 提交与 PR

- 尽量保持提交小而清晰，包含必要的测试/说明。
- PR 描述里写清楚：动机、变更点、如何验证。
- 如果变更涉及存储路径/配置项，请同步更新 `README.md` 和 `mcp.json.example`。

