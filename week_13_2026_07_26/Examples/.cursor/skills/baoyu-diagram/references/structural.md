# 结构图布局

涵盖：类图、ER 图、组件图、包图、组织架构图。

## 类图

### 类方框（三格）

```svg
<g transform="translate(X, Y)">
  <!-- 遮罩 -->
  <rect width="180" height="120" rx="6" fill="#0f172a"/>
  <!-- 方框 -->
  <rect width="180" height="120" rx="6" fill="rgba(8,51,68,0.4)" stroke="#22d3ee" stroke-width="1.5"/>
  <!-- 类名格 -->
  <text x="90" y="24" fill="white" font-size="11" font-weight="700" text-anchor="middle">ClassName</text>
  <!-- 分隔线 1 -->
  <line x1="0" y1="35" x2="180" y2="35" stroke="#22d3ee" stroke-width="0.5" stroke-opacity="0.5"/>
  <!-- 属性 -->
  <text x="10" y="52" fill="#94a3b8" font-size="8">- id: int</text>
  <text x="10" y="64" fill="#94a3b8" font-size="8">- name: string</text>
  <!-- 分隔线 2 -->
  <line x1="0" y1="75" x2="180" y2="75" stroke="#22d3ee" stroke-width="0.5" stroke-opacity="0.5"/>
  <!-- 方法 -->
  <text x="10" y="92" fill="#94a3b8" font-size="8">+ getName(): string</text>
  <text x="10" y="104" fill="#94a3b8" font-size="8">+ setName(s: string)</text>
</g>
```

抽象类把类名设为斜体。接口在名称上方用较小字体加 `«interface»`。

### 关系线

| 关系 | 线型 | 箭头/端点 |
|------|------|---------|
| 继承（Inheritance） | 实线 | 指向父类的空心三角（▷） |
| 实现（Implementation） | 虚线 | 指向接口的空心三角 |
| 组合（Composition） | 实线 | 拥有端为实心菱形（◆） |
| 聚合（Aggregation） | 实线 | 拥有端为空心菱形（◇） |
| 依赖（Dependency） | 虚线 | 依赖目标端为开放箭头 |
| 关联（Association） | 实线 | 开放箭头或无 |

**标记：**

```svg
<!-- 继承三角 -->
<marker id="inherit" markerWidth="12" markerHeight="10" refX="12" refY="5" orient="auto">
  <polygon points="0 0, 12 5, 0 10" fill="#0f172a" stroke="#94a3b8" stroke-width="1.5"/>
</marker>

<!-- 组合菱形 -->
<marker id="composition" markerWidth="12" markerHeight="8" refX="0" refY="4" orient="auto">
  <polygon points="0 4, 6 0, 12 4, 6 8" fill="#94a3b8"/>
</marker>

<!-- 聚合菱形 -->
<marker id="aggregation" markerWidth="12" markerHeight="8" refX="0" refY="4" orient="auto">
  <polygon points="0 4, 6 0, 12 4, 6 8" fill="#0f172a" stroke="#94a3b8" stroke-width="1.5"/>
</marker>
```

### 基数标签

放在关系线两端，距方框边缘偏移 5-8px：

```svg
<text x="X" y="Y" fill="#94a3b8" font-size="8">1..*</text>
```

## ER 图

与类图类似，但：
- 用两格方框（实体名 + 属性）
- 主键用 `PK` 前缀并加粗
- 外键用 `FK` 前缀
- 关系线用鸦爪记法：

```svg
<!-- 一端（单线） -->
<line x1="X1" y1="Y" x2="X1+15" y2="Y" stroke="#94a3b8" stroke-width="1.5"/>
<!-- 多端（鸦爪） -->
<line x1="X2-15" y1="Y-6" x2="X2" y2="Y" stroke="#94a3b8" stroke-width="1.5"/>
<line x1="X2-15" y1="Y+6" x2="X2" y2="Y" stroke="#94a3b8" stroke-width="1.5"/>
<line x1="X2-15" y1="Y" x2="X2" y2="Y" stroke="#94a3b8" stroke-width="1.5"/>
```

## 组织架构图

- 自上而下的树形布局
- 根节点在顶部居中
- 每层等距分布（垂直间距 100-120px）
- 同级节点水平均匀分布
- 连接线：从父节点底部中心向下到一根横条，再从横条向下到每个子节点顶部中心
- 用颜色区分部门或层级

## 布局技巧

- 先数最宽的那一层，以确定图表总宽度
- 在 viewBox 中水平居中树形
- 对很深的树（5 层以上），考虑改用横向布局
