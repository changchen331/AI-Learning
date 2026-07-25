"""
备份与恢复工具 — 用于测试后回到初始演示状态

使用方式：
  python src/reset.py backup           # 创建当前状态快照（自动以时间戳命名）
  python src/reset.py restore          # 恢复最近一次快照
  python src/reset.py restore --list   # 列出所有快照
  python src/reset.py restore <名称>   # 恢复指定快照（如 20260508_143022）
  python src/reset.py factory          # 回到出厂初始态（空记忆、空数据库）

备份内容（运行时产生的状态文件）：
  memory/USER.md          用户画像（Memory Flush 自动更新）
  memory/MEMORY.md        跨会话记忆条目
  data/vector_index/      FAISS 向量索引
  outputs/sessions/       SQLite 会话数据库
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKUP_DIR = ROOT / "backups"

# 需要备份的路径（相对于 ROOT）
BACKUP_TARGETS = [
    "memory/USER.md",
    "memory/MEMORY.md",
    "memory/HEARTBEAT.md",  # 调度意图检测会自动写入，需要纳入备份
    "memory/SOUL.md",  # /reset 时从 initial 快照恢复
    "memory/AGENTS.md",  # /reset 时从 initial 快照恢复
    "data/vector_index",
    "outputs/sessions",
]

# 出厂初始内容
INITIAL_USER_MD = """\
# USER.md — 用户偏好与已知信息

> 本文件由 Memory Flush 流程自动维护，也可手动编辑。
> 最后更新：（尚未初始化）

## 基本信息
- 姓名：（尚未告知）
- 所在地：（尚未告知）
- 职业：（尚未告知）

## 偏好
（暂无记录，对话后由 Memory Flush 自动填充）

## 技术背景
（暂无记录）

## 沟通偏好
（暂无记录）

## 其他已知信息
（暂无记录）
"""

INITIAL_MEMORY_MD = """\
# MEMORY.md — 跨会话持久记忆

> 本文件由 Memory Flush 流程自动维护。
> 每条记忆格式：`### [类别] 标题` + 记录时间 + 内容
> 类别：preference（偏好）| fact（事实）| event（事件）| decision（决策）

<!-- MEMORY_ENTRIES_START -->
<!-- MEMORY_ENTRIES_END -->
"""

INITIAL_HEARTBEAT_MD = """\
# HEARTBEAT.md — 自动化任务脑

> 本文件定义 Agent 的定时自动任务。
> 任务条目由对话中检测到调度意图时自动写入，也可手动编辑。
> 修改本文件后，调度器会在下一分钟内自动重新加载。

## 格式说明

每个任务块以 `### TASK: {name}` 开头，包含以下字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| trigger | 标准 5 字段 cron 表达式（分 时 日 月 周）| `0 9 * * 1-5` |
| enabled | 是否启用 | `true` / `false` |
| action | 执行动作，见下方支持列表 | `send_message` |
| description | 任务说明（供人类和 LLM 读取）| 工作日早上问候 |
| prompt | 仅 send_message 动作需要，LLM 生成消息时使用的提示 | （可选）|
| added | 写入时间 | `2026-05-08` |

## 支持的 action 类型

| action | 说明 |
|--------|------|
| `send_message` | LLM 根据 prompt 和用户画像生成一条主动消息，推送到前端 |
| `summarize_sessions` | 汇总近期对话，写入 MEMORY.md [event] 条目 |
| `compact_memory` | 触发 Memory Compaction，压缩旧记忆条目 |
| `user_profile_refresh` | 重新分析全部记忆，刷新 USER.md |

---

## 已配置任务

<!-- TASKS_START -->
### TASK: weekly_compaction
trigger: 0 3 * * 0
enabled: true
action: compact_memory
description: 每周日凌晨3点自动压缩旧记忆条目
added: 2026-05-08

<!-- TASKS_END -->
"""


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _snapshot_dir(name: str) -> Path:
    return BACKUP_DIR / name


def _list_snapshots() -> list[str]:
    if not BACKUP_DIR.exists():
        return []
    names = [d.name for d in BACKUP_DIR.iterdir() if d.is_dir()]
    # initial 固定排在最后，其余按时间戳倒序
    timestamps = sorted([n for n in names if n != "initial"], reverse=True)
    return timestamps + (["initial"] if "initial" in names else [])


def _copy_target(src_root: Path, dst_root: Path, rel: str):
    src = src_root / rel
    dst = dst_root / rel
    if not src.exists():
        return
    if src.is_dir():
        # 逐文件复制，对被锁定的 .db 文件使用 SQLite API 恢复
        dst.mkdir(parents=True, exist_ok=True)
        for src_file in src.rglob("*"):
            if src_file.is_dir():
                continue
            dst_file = dst / src_file.relative_to(src)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if src_file.suffix == ".db":
                _restore_sqlite(src_file, dst_file)
            else:
                try:
                    shutil.copy2(src_file, dst_file)
                except PermissionError:
                    pass  # 非关键文件，跳过
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _restore_sqlite(src_db: Path, dst_db: Path):
    """将 src_db 的数据恢复到 dst_db（支持 dst_db 被其他进程占用的情况）"""
    import sqlite3

    # 读取备份数据
    src_conn = sqlite3.connect(src_db)
    sessions = src_conn.execute("SELECT * FROM sessions").fetchall() if _table_exists(src_conn, "sessions") else []
    messages = src_conn.execute("SELECT * FROM messages").fetchall() if _table_exists(src_conn, "messages") else []
    src_conn.close()

    # 写入目标（清空后写入）
    dst_conn = sqlite3.connect(dst_db)
    dst_conn.executescript("""
                           CREATE TABLE IF NOT EXISTS sessions
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               start_time
                               TEXT
                               NOT
                               NULL,
                               end_time
                               TEXT,
                               title
                               TEXT,
                               flushed
                               INTEGER
                               DEFAULT
                               0
                           );
                           CREATE TABLE IF NOT EXISTS messages
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               session_id
                               INTEGER
                               NOT
                               NULL,
                               role
                               TEXT
                               NOT
                               NULL,
                               content
                               TEXT
                               NOT
                               NULL,
                               timestamp
                               TEXT
                               NOT
                               NULL,
                               FOREIGN
                               KEY
                           (
                               session_id
                           ) REFERENCES sessions
                           (
                               id
                           )
                               );
                           DELETE
                           FROM messages;
                           DELETE
                           FROM sessions;
                           """)
    if sessions:
        dst_conn.executemany("INSERT INTO sessions VALUES (?,?,?,?,?)", sessions)
    if messages:
        dst_conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?)", messages)
    dst_conn.commit()
    dst_conn.close()


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ── 操作函数 ──────────────────────────────────────────────────────────────────

def cmd_backup(name: str = None) -> str:
    """保存当前状态快照"""
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = _snapshot_dir(name)
    snap_dir.mkdir(parents=True, exist_ok=True)

    backed = []
    for rel in BACKUP_TARGETS:
        src = ROOT / rel
        if src.exists():
            _copy_target(ROOT, snap_dir, rel)
            backed.append(rel)

    print(f"快照已保存：backups/{name}/")
    for b in backed:
        print(f"  · {b}")
    if not backed:
        print("  （无运行时状态文件，快照为空）")
    return name


def cmd_restore(name: str = None):
    """从快照恢复"""
    if name == "--list":
        snaps = _list_snapshots()
        if not snaps:
            print("暂无快照。使用 `python src/reset.py backup` 创建。")
        else:
            print(f"共 {len(snaps)} 个快照（最新在前）：")
            for s in snaps:
                print(f"  · {s}")
        return

    if name is None:
        snaps = _list_snapshots()
        if not snaps:
            print("暂无快照。请先 backup 或使用 factory 回到出厂态。")
            sys.exit(1)
        name = snaps[0]
        print(f"使用最近快照：{name}")

    snap_dir = _snapshot_dir(name)
    if not snap_dir.exists():
        print(f"快照不存在：backups/{name}/")
        sys.exit(1)

    restored = []
    for rel in BACKUP_TARGETS:
        src = snap_dir / rel
        if src.exists():
            _copy_target(snap_dir, ROOT, rel)
            restored.append(rel)

    print(f"已从 backups/{name}/ 恢复：")
    for r in restored:
        print(f"  · {r}")
    if not restored:
        print("  （快照为空，未恢复任何文件）")


def cmd_factory():
    """回到出厂初始态：清空记忆、清空数据库"""
    # 写回初始 Markdown
    (ROOT / "memory" / "USER.md").write_text(INITIAL_USER_MD, encoding="utf-8")
    (ROOT / "memory" / "MEMORY.md").write_text(INITIAL_MEMORY_MD, encoding="utf-8")
    (ROOT / "memory" / "HEARTBEAT.md").write_text(INITIAL_HEARTBEAT_MD, encoding="utf-8")
    print("memory/USER.md      → 已重置为初始模板")
    print("memory/MEMORY.md    → 已重置为空（无记忆条目）")
    print("memory/HEARTBEAT.md → 已重置为初始任务配置")

    # 清空 FAISS 索引
    index_dir = ROOT / "data" / "vector_index"
    for f in ["memory.faiss", "memory_meta.pkl"]:
        p = index_dir / f
        if p.exists():
            p.unlink()
    print("data/vector_index/ → FAISS 索引已清空")

    # 清空 SQLite（优先删除文件；被占用时清空表数据）
    db_path = ROOT / "outputs" / "sessions" / "memory.db"
    if db_path.exists():
        try:
            db_path.unlink()
            print("outputs/sessions/  → SQLite 数据库已清空")
        except PermissionError:
            # 文件被占用（如 serve.py 正在运行），改为清空表数据
            conn = sqlite3.connect(db_path)
            conn.executescript("DELETE FROM messages; DELETE FROM sessions;")
            conn.commit()
            conn.close()
            print("outputs/sessions/  → SQLite 数据已清空（文件保留，因被其他进程占用）")
    else:
        print("outputs/sessions/  → SQLite 数据库不存在，跳过")

    print("\n出厂初始态恢复完成，可以重新演示了。")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    if cmd == "backup":
        name = args[1] if len(args) > 1 else None
        cmd_backup(name)
    elif cmd == "restore":
        name = args[1] if len(args) > 1 else None
        cmd_restore(name)
    elif cmd == "factory":
        cmd_factory()
    else:
        print(f"未知命令：{cmd}")
        print("用法：python src/reset.py [backup|restore|factory]")
        sys.exit(1)


if __name__ == "__main__":
    main()
