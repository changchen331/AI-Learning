/* =============================================================================
 *  topology.js  ——  拓扑可视化（辅助代码，非教学重点）· 深色科技感
 * =============================================================================
 *  vanillar JS + SVG，零依赖。学生不需要读懂这里。
 *  改动：深色背景 + 节点发光 + 运行节点脉冲 + 边流光 + 正确 reset（切换问题整体换图）
 * =========================================================================== */
class TopoViz {
    constructor(host) {
        this.host = host;
        this.host.innerHTML = '';          // ← 切换问题整体换图，不堆叠
        this.svgNS = 'http://www.w3.org/2000/svg';
        this.subs = {};
        this.order = [];
        this._clickCb = null;
        this.W = 380;
        this.H = 480;
        this.mainXY = {x: this.W / 2, y: 44};
        this.svg = this._svg();
        this.host.appendChild(this.svg);
        // 滤镜：发光
        const defs = document.createElementNS(this.svgNS, 'defs');
        defs.innerHTML = `
      <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="3.5" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>`;
        this.svg.appendChild(defs);
    }

    _svg() {
        const s = document.createElementNS(this.svgNS, 'svg');
        s.setAttribute('viewBox', `0 0 ${this.W} ${this.H}`);
        s.setAttribute('width', '100%');
        s.setAttribute('style', 'background:radial-gradient(circle at 50% 30%, #142042 0%, #070b16 80%);border-radius:8px');
        return s;
    }

    _node(x, y, r, fill, label, id, glowColor) {
        const g = document.createElementNS(this.svgNS, 'g');
        g.style.cursor = 'pointer';
        const c = document.createElementNS(this.svgNS, 'circle');
        c.setAttribute('cx', x);
        c.setAttribute('cy', y);
        c.setAttribute('r', r);
        c.setAttribute('fill', fill);
        c.setAttribute('stroke', glowColor || '#00d4ff');
        c.setAttribute('stroke-width', '2');
        c.setAttribute('filter', 'url(#glow)');
        c.style.transition = 'all .3s';
        const t = document.createElementNS(this.svgNS, 'text');
        t.setAttribute('x', x);
        t.setAttribute('y', y + r + 13);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('font-size', '8.5');
        t.setAttribute('fill', '#8aa6d0');
        t.textContent = label;
        g.appendChild(c);
        g.appendChild(t);
        if (id) g.addEventListener('click', () => this._clickCb && this._clickCb(id));
        this.svg.appendChild(g);
        return {g, c, t};
    }

    _edge(x1, y1, x2, y2, color) {
        const ln = document.createElementNS(this.svgNS, 'line');
        ln.setAttribute('x1', x1);
        ln.setAttribute('y1', y1);
        ln.setAttribute('x2', x2);
        ln.setAttribute('y2', y2);
        ln.setAttribute('stroke', color || '#1f3a6b');
        ln.setAttribute('stroke-width', '1.5');
        ln.setAttribute('stroke-dasharray', '4 4');
        this.svg.appendChild(ln);
        return ln;
    }

    setMain() {
        const o = this._node(this.mainXY.x, this.mainXY.y, 18, '#0a2a5e', '主 agent', 'main', '#00d4ff');
        this.subs['main'] = {...o, x: this.mainXY.x, y: this.mainXY.y, status: 'idle'};
    }

    addSubagent(id, topic) {
        const i = this.order.length;
        // 网格布局：2 列
        const col = i % 2, row = Math.floor(i / 2);
        const x = 80 + col * 220;
        const y = 150 + row * 110;
        const o = this._node(x, y, 14, '#103a5c',
            topic.length > 9 ? topic.slice(0, 9) + '…' : topic, id, '#4f8cff');
        // 主→子 边
        this._edge(this.mainXY.x, this.mainXY.y + 18, x, y - 14);
        this.subs[id] = {...o, x, y, status: 'idle', topic};
        this.order.push(id);
    }

    markRunning(id) {
        const s = this.subs[id];
        if (!s) return;
        s.status = 'running';
        s.c.setAttribute('stroke', '#ffb020');
        s.c.setAttribute('stroke-width', '3.5');
        s.c.setAttribute('fill', '#3a2a00');
        // 脉冲：半径周期变化
        if (!s._pulse) {
            s._pulse = setInterval(() => {
                if (s.status !== 'running') {
                    clearInterval(s._pulse);
                    s._pulse = null;
                    return;
                }
                s.c.setAttribute('r', s.c.getAttribute('r') === '16' ? '13' : '16');
            }, 450);
        }
    }

    markDone(id) {
        const s = this.subs[id];
        if (!s) return;
        s.status = 'done';
        if (s._pulse) {
            clearInterval(s._pulse);
            s._pulse = null;
        }
        s.c.setAttribute('fill', '#0d3d2a');
        s.c.setAttribute('stroke', '#2ee6a0');
        s.c.setAttribute('stroke-width', '2.5');
        s.c.setAttribute('r', '14');
    }

    reset() {
        Object.values(this.subs).forEach(s => {
            if (s._pulse) clearInterval(s._pulse);
        });
        this.host.innerHTML = '';
        this.subs = {};
        this.order = [];
        // 重建空 svg 容器（下次 new TopoViz 会重新建）
    }

    onClick(cb) {
        this._clickCb = cb;
    }
}
