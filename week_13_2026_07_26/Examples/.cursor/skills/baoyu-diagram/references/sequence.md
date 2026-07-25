# 时序图布局

## 核心元素

| 元素 | 视觉 | 说明 |
|------|------|------|
| 角色/参与者 | 顶部方框 + 垂直虚线生命线 | 交互中的每个实体 |
| 同步消息 | 实线箭头 → | 请求或调用 |
| 异步消息 | 开放箭头 → | 触发即忘（fire-and-forget） |
| 返回消息 | 虚线箭头 ← | 响应 |
| 激活条 | 生命线上的窄填充矩形 | 实体正在处理 |
| 自消息 | 回到同一生命线的箭头 | 内部处理 |
| 备注 | 带折角的圆角矩形 | 注释 |
| Alt/Opt 框 | 带标签页的虚线边界 | 条件块 |
| Loop 框 | 带 "loop" 标签页的虚线边界 | 循环 |

## 布局算法

1. **放置角色** 横向排列在顶部，等距分布（间距 150-200px）
2. **画生命线** 从每个角色方框向下画垂直虚线
3. **放置消息** 作为生命线之间的横向箭头，按时间顺序自上而下
4. **消息间垂直间距：** 40-50px
5. **激活条：** 宽 10px，居中于生命线，从入站消息延伸到出站消息

## 角色方框

```svg
<!-- 角色方框 -->
<rect x="X" y="20" width="130" height="45" rx="6" fill="#0f172a"/>
<rect x="X" y="20" width="130" height="45" rx="6" fill="rgba(8,51,68,0.4)" stroke="#22d3ee" stroke-width="1.5"/>
<text x="CX" y="47" fill="white" font-size="11" font-weight="600" text-anchor="middle">Actor Name</text>

<!-- 生命线 -->
<line x1="CX" y1="65" x2="CX" y2="BOTTOM" stroke="#334155" stroke-width="1" stroke-dasharray="6,4"/>
```

## 消息箭头

```svg
<!-- 同步消息（实线箭头） -->
<line x1="FROM_CX" y1="Y" x2="TO_CX" y2="Y" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
<text x="MID_X" y="Y-8" fill="#e2e8f0" font-size="9" text-anchor="middle">methodCall()</text>

<!-- 返回消息（虚线箭头，反向） -->
<line x1="TO_CX" y1="Y" x2="FROM_CX" y2="Y" stroke="#64748b" stroke-width="1" stroke-dasharray="6,3" marker-end="url(#arrow)"/>
<text x="MID_X" y="Y-8" fill="#94a3b8" font-size="8" text-anchor="middle" font-style="italic">response</text>

<!-- 自消息（回环箭头） -->
<path d="M CX,Y L CX+40,Y L CX+40,Y+25 L CX,Y+25" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
<text x="CX+45" y="Y+15" fill="#e2e8f0" font-size="8">process()</text>
```

## 激活条

```svg
<rect x="CX-5" y="START_Y" width="10" height="H" rx="2" fill="rgba(8,51,68,0.6)" stroke="#22d3ee" stroke-width="1"/>
```

## 条件 / 循环框

```svg
<!-- 框边界 -->
<rect x="X" y="Y" width="W" height="H" rx="4" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="4,3"/>
<!-- 框标签页 -->
<rect x="X" y="Y" width="50" height="18" rx="4" fill="rgba(30,41,59,0.8)" stroke="#64748b" stroke-width="1"/>
<text x="X+25" y="Y+13" fill="#94a3b8" font-size="8" font-weight="600" text-anchor="middle">alt</text>
<!-- 条件文本 -->
<text x="X+60" y="Y+13" fill="#94a3b8" font-size="8" font-style="italic">[condition]</text>
<!-- else 分隔线 -->
<line x1="X" y1="MID_Y" x2="X+W" y2="MID_Y" stroke="#64748b" stroke-width="1" stroke-dasharray="4,3"/>
<text x="X+10" y="MID_Y+13" fill="#94a3b8" font-size="8" font-style="italic">[else]</text>
```

## 编号

对复杂时序（8 条以上消息），给每条消息编号：

```svg
<circle cx="FROM_CX-15" cy="Y" r="8" fill="rgba(59,130,246,0.3)" stroke="#60a5fa" stroke-width="1"/>
<text x="FROM_CX-15" y="Y+3" fill="#60a5fa" font-size="7" font-weight="600" text-anchor="middle">1</text>
```

## 颜色分配

从调色板为每个角色分配一种独特颜色。该颜色用于：
- 角色方框描边
- 该生命线上的激活条
- 从该角色发出的箭头（可选，用于在复杂图中提升可读性）
