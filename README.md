# Memvid MCP

基于 [memvid](https://github.com/Olow304/memvid) 的 MCP 服务器，提供类似 OpenMemory 的认知记忆存储功能。

## 特性

### 双记忆系统
- **项目记忆** (`.memvid_data/`) - 项目特定的知识和决策
- **用户记忆** (`~/memvid_data/`) - 个人偏好和通用知识
- **语义分类** - 自动判断记忆属于项目还是用户

### 认知记忆模型
- **五扇区分类** - episodic/semantic/procedural/emotional/reflective
- **时间知识图谱** - 支持时间点查询的事实存储
- **Waypoint 关联图** - 记忆之间的语义关联
- **记忆衰减** - 基于时间和使用频率的自然遗忘

## 安装

```bash
# 使用 uv
uv pip install -e .

# 或使用 pip
pip install -e .
```

## 使用

### 作为 MCP 服务器

添加到 Claude Code 配置 (`~/.claude.json`):

```json
{
  "mcpServers": {
    "memvid": {
      "command": "memvid-mcp"
    }
  }
}
```

#### 数据目录与环境变量

默认存储位置：

- 项目记忆：`<project_root>/.memvid_data/`（项目根目录会自动通过 `.git`、`pyproject.toml` 等标记检测）
- 用户记忆：`~/memvid_data/`

可选环境变量（启动 MCP 服务器时设置）：

- `MEMVID_PROJECT_DATA_DIR`：覆盖项目记忆目录（相对路径以项目根目录为基准）
- `MEMVID_USER_DATA_DIR`：覆盖用户记忆目录（支持 `~`；相对路径以 `~` 为基准）
- `MEMVID_PROJECT_ROOT`：强制指定项目根目录（从非项目目录启动时有用）
- `MEMVID_DATA_DIR`：兼容别名，等同于 `MEMVID_PROJECT_DATA_DIR`

示例见 `mcp.json.example`。

### MCP 工具

#### 核心记忆操作
- `memvid_store` - 存储记忆（自动分类 scope 和 sector）
- `memvid_query` - 语义搜索（合并项目和用户记忆）
- `memvid_get` - 按 ID 获取记忆
- `memvid_list` - 列出记忆
- `memvid_delete` - 删除记忆
- `memvid_stats` - 获取统计信息

#### 时间知识图谱
- `memvid_store_fact` - 存储时间事实 (subject, predicate, object)
- `memvid_query_facts` - 时间点查询
- `memvid_get_timeline` - 获取实体时间线

#### 衰减与强化
- `memvid_reinforce` - 手动强化记忆
- `memvid_apply_decay` - 应用时间衰减

### Python API

```python
from memvid_mcp import DualMemoryManager

# 初始化双记忆管理器
manager = DualMemoryManager()

# 存储记忆（自动分类）
result = manager.store("This project uses FastAPI for REST APIs")
# → 自动存储到项目记忆

result = manager.store("I prefer pytest over unittest")
# → 自动存储到用户记忆

# 手动指定 scope
result = manager.store("Bug in auth.py line 42", scope="project")

# 搜索记忆（合并两个库的结果）
results = manager.recall("testing framework", limit=10)

# 获取统计
stats = manager.stats()
print(stats["project"]["total_memories"])
print(stats["user"]["total_memories"])
```

### Claude Code 技能

项目包含 Claude Code 技能定义 (`skills/memvid-core/SKILL.md`)，支持自然语言交互：

- "记住这个项目使用 FastAPI"
- "我喜欢用 pytest 做测试"
- "关于测试你记得什么？"
- "显示记忆统计"

## 架构

```
memvid-mcp/
├── src/memvid_mcp/
│   ├── server.py          # MCP 服务器
│   ├── memory.py          # 核心记忆类
│   ├── dual_memory.py     # 双记忆管理器
│   ├── scope_classifier.py # 语义范围分类器
│   ├── classifier.py      # 认知扇区分类器
│   ├── temporal.py        # 时间知识图谱
│   ├── waypoint.py        # 关联图
│   └── decay.py           # 衰减算法
├── skills/
│   └── memvid-core/       # Claude Code 技能
└── tests/                 # 测试套件 (90 个测试)
```

## 与 OpenMemory 的对比

| 功能 | OpenMemory | Memvid MCP |
|------|------------|------------|
| 存储后端 | SQLite/Postgres + 向量 | MP4 视频 + JSON 索引 |
| 语义搜索 | 多扇区向量搜索 | FAISS 向量搜索 |
| 用户隔离 | ✅ | ✅ |
| 双记忆系统 | ❌ | ✅ (项目 + 用户) |
| 时间知识图谱 | ✅ | ✅ |
| Waypoint 图 | ✅ | ✅ |
| 记忆衰减 | ✅ | ✅ |
| Claude Code 技能 | ❌ | ✅ |

## 许可证

MIT
