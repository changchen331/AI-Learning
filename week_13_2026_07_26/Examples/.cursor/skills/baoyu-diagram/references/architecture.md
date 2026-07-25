# 架构图布局

## 流向

选择一个主方向：
- **从左到右（LTR）：** 适合数据管线、请求流。用户/客户端在左，数据存储在右。
- **从上到下（TTB）：** 适合分层架构。客户端在顶部，基础设施在底部。

## 布局算法

1. **识别层次：** 按角色分组组件（客户端、网关、服务、数据、基础设施）
2. **分配列（LTR）或行（TTB）：** 每层一列/一行
3. **层内：** 在 LTR 下纵向堆叠组件（或在 TTB 下横向排列），最小间距 40px
4. **区域边界：** 围绕共享基础设施的分组绘制（如 "AWS us-east-1"、"Kubernetes Cluster"）
5. **连接器：** 在层之间布线箭头。对于层间的总线/队列，在间隙中放一根细连接条。

## 典型分层结构（LTR）

```
Col 1 (x=40)     Col 2 (x=250)     Col 3 (x=460)     Col 4 (x=670)
┌──────────┐     ┌──────────┐      ┌──────────┐      ┌──────────┐
│  Client   │────▶│ Gateway  │─────▶│ Services │─────▶│ Database │
│  Layer    │     │  Layer   │      │  Layer   │      │  Layer   │
└──────────┘     └──────────┘      └──────────┘      └──────────┘
```

列间距：列起点之间 200-220px。组件更宽时相应调整。

## 典型分层结构（TTB）

```
Row 1 (y=60):   [ Browser ]  [ Mobile App ]  [ API Client ]
Row 2 (y=160):  [         Load Balancer / API Gateway       ]
Row 3 (y=280):  [ Auth Svc ]  [ User Svc ]  [ Order Svc ]
Row 4 (y=400):  [  Redis  ]   [ PostgreSQL ]  [ S3 Bucket ]
```

行间距：行起点之间 120-140px。

## 连接布线

- 优先使用笔直的横线或竖线
- 会穿过组件的连接，用两段式（L 形）路径：
  ```svg
  <path d="M x1,y1 L midX,y1 L midX,y2" fill="none" stroke="#64748b" marker-end="url(#arrow)"/>
  ```
- 繁忙图表中，对次要连接使用 `stroke-opacity="0.6"`
- 在中点附近用 text 元素标注重要连接

## 消息总线 / 事件总线模式

当服务通过共享总线通信时，在服务层之间画一根横向总线：

```
Services:  [ Svc A ]    [ Svc B ]    [ Svc C ]
              │              │            │
Bus:     ════╪══════════════╪════════════╪═══════
              │              │            │
Data:    [ DB A ]        [ DB B ]     [ Cache ]
```

总线用 Connector 色（橙色）。

## 多区域 / 多云

嵌套区域边界：
- 外层边界：云厂商（AWS、GCP）
- 内层边界：区域或 VPC
- 最内层：可用区或子网

用不同的虚线模式区分嵌套层级：
- 外层：`stroke-dasharray="12,4"`
- 中层：`stroke-dasharray="8,4"`
- 内层：`stroke-dasharray="4,4"`
