# HEARTBEAT.md — 自动化任务脑

> 本文件定义 Agent 的定时自动任务。
> 任务条目由对话中检测到调度意图时自动写入，也可手动编辑。
> 修改本文件后，调度器会在下一分钟内自动重新加载。

## 格式说明

每个任务块以 `### TASK: {name}` 开头，包含以下字段：

| 字段        | 说明                                               | 示例             |
|-------------|----------------------------------------------------|------------------|
| trigger     | 标准 5 字段 cron 表达式（分 时 日 月 周）          | `0 9 * * 1-5`    |
| enabled     | 是否启用                                           | `true` / `false` |
| action      | 执行动作，见下方支持列表                           | `send_message`   |
| description | 任务说明（供人类和 LLM 读取）                      | 工作日早上问候   |
| prompt      | 仅 send_message 动作需要，LLM 生成消息时使用的提示 | （可选）         |
| added       | 写入时间                                           | `2026-05-08`     |

## 支持的 action 类型

| action                 | 说明                                                   |
|------------------------|--------------------------------------------------------|
| `send_message`         | LLM 根据 prompt 和用户画像生成一条主动消息，推送到前端 |
| `summarize_sessions`   | 汇总近期对话，写入 MEMORY.md [event] 条目              |
| `compact_memory`       | 触发 Memory Compaction，压缩旧记忆条目                 |
| `user_profile_refresh` | 重新分析全部记忆，刷新 USER.md                         |

---

## 已配置任务

<!-- TASKS_START -->

### TASK: weekly_compaction

trigger: 0 3 * * 0 enabled: true action: compact_memory description: 每周日凌晨3点自动压缩旧记忆条目 added: 2026-05-08

### TASK: check_flight_reminder

trigger: * * * * *
enabled: false action: send_message description: 每分钟提醒用户查询北京到上海的机票 prompt:
请提醒用户：现在需要查询北京到上海的机票信息，请检查是否有可用的航班。 added: 2026-05-08

<!-- TASKS_END -->
