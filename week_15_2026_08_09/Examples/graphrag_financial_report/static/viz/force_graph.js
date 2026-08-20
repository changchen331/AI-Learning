/* =============================================================================
 *  force_graph.js  ——  纯手写 force-directed 图可视化（辅助代码，非教学重点）
 * =============================================================================
 *  这部分是「让学生看清 GraphRAG 子图长什么样」的可视化辅助，与主流程无关。
 *  学生不需要读懂这里。无任何外部依赖，vanilla JS + Canvas。
 *
 *  用法（在 index.html 里）：
 *    const fg = new ForceGraph(document.getElementById('graphCanvas'));
 *    fg.setData(nodes, edges);   // nodes:[{uid,name,type}], edges:[{src,dst,rel}]
 *    fg.start();
 *    fg.stop();
 * =========================================================================== */
class ForceGraph {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.edges = [];
        this.raf = null;
        // 实体类型 → 颜色（与页面 step 卡片配色一致）
        this.colors = {
            Company: '#1E3A5F', Person: '#2E86AB', Subsidiary: '#D4720A',
            Product: '#217B45', Indicator: '#C0392B', Segment: '#7D3C98',
            Region: '#117A65', default: '#666'
        };
        // 拖拽交互
        this.dragNode = null;
        canvas.addEventListener('mousedown', e => this._onDown(e));
        canvas.addEventListener('mousemove', e => this._onMove(e));
        canvas.addEventListener('mouseup', () => this.dragNode = null);
        canvas.addEventListener('mouseleave', () => this.dragNode = null);
    }

    setData(nodes, edges) {
        const w = this.canvas.width, h = this.canvas.height;
        this.nodes = nodes.map(n => ({
            ...n,
            x: w / 2 + (Math.random() - 0.5) * 200,
            y: h / 2 + (Math.random() - 0.5) * 200,
            vx: 0, vy: 0, r: 12
        }));
        // 边用 uid 连接（兼容 src/dst 两种字段名）
        const byName = {};
        this.nodes.forEach(n => byName[n.uid] = n);
        this.edges = edges.map(e => ({
            a: byName[e.src] || byName[e.src_uid],
            b: byName[e.dst] || byName[e.dst_uid],
            rel: e.rel || e.type
        })).filter(e => e.a && e.b);
    }

    _step() {
        // 简化版 force：节点间斥力 + 边弹簧 + 向中心收拢
        const k_rep = 1800, k_spring = 0.04, k_center = 0.005, len = 80;
        const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
        // 斥力
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const a = this.nodes[i], b = this.nodes[j];
                let dx = a.x - b.x, dy = a.y - b.y;
                let d2 = dx * dx + dy * dy + 0.01;
                let f = k_rep / d2;
                let dn = Math.sqrt(d2);
                a.vx += (dx / dn) * f;
                a.vy += (dy / dn) * f;
                b.vx -= (dx / dn) * f;
                b.vy -= (dy / dn) * f;
            }
        }
        // 弹簧
        this.edges.forEach(e => {
            let dx = e.b.x - e.a.x, dy = e.b.y - e.a.y;
            let dn = Math.sqrt(dx * dx + dy * dy) + 0.01;
            let f = k_spring * (dn - len);
            e.a.vx += (dx / dn) * f;
            e.a.vy += (dy / dn) * f;
            e.b.vx -= (dx / dn) * f;
            e.b.vy -= (dy / dn) * f;
        });
        // 收拢 + 阻尼 + 移动
        this.nodes.forEach(n => {
            if (n === this.dragNode) return;
            n.vx += (cx - n.x) * k_center;
            n.vy += (cy - n.y) * k_center;
            n.vx *= 0.85;
            n.vy *= 0.85;
            n.x += n.vx;
            n.y += n.vy;
        });
        this._draw();
    }

    _draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // 边
        ctx.strokeStyle = '#bbb';
        ctx.lineWidth = 1;
        ctx.font = '9px Calibri';
        ctx.fillStyle = '#888';
        this.edges.forEach(e => {
            ctx.beginPath();
            ctx.moveTo(e.a.x, e.a.y);
            ctx.lineTo(e.b.x, e.b.y);
            ctx.stroke();
            // 关系标签
            const mx = (e.a.x + e.b.x) / 2, my = (e.a.y + e.b.y) / 2;
            if (e.rel) ctx.fillText(e.rel, mx, my);
        });
        // 节点
        this.nodes.forEach(n => {
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r, 0, 2 * Math.PI);
            ctx.fillStyle = this.colors[n.type] || this.colors.default;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.stroke();
            // 名字
            ctx.fillStyle = '#333';
            ctx.font = '11px Calibri';
            ctx.textAlign = 'center';
            ctx.fillText(n.name.length > 8 ? n.name.slice(0, 8) + '…' : n.name, n.x, n.y + n.r + 12);
        });
    }

    _onDown(e) {
        const r = this.canvas.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        this.dragNode = this.nodes.find(n => Math.hypot(n.x - mx, n.y - my) < n.r) || null;
    }

    _onMove(e) {
        if (!this.dragNode) return;
        const r = this.canvas.getBoundingClientRect();
        this.dragNode.x = e.clientX - r.left;
        this.dragNode.y = e.clientY - r.top;
        this.dragNode.vx = 0;
        this.dragNode.vy = 0;
    }

    start() {
        this._loop();
    }

    _loop() {
        this._step();
        this.raf = requestAnimationFrame(() => this._loop());
    }

    stop() {
        if (this.raf) cancelAnimationFrame(this.raf);
    }

    clear() {
        this.stop();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.nodes = [];
        this.edges = [];
    }
}
