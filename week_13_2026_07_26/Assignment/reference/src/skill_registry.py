"""
Skill 注册中心 —— 渐进式加载的两阶段实现

教学重点：
  1. 预加载（Phase 1）：只读每个 SKILL.md 的 frontmatter，拼成轻量 meta 列表注入系统提示，
     让 LLM 先「选 skill」而无需看到完整说明，大幅省 token。
  2. 按需加载（Phase 2 / 懒加载）：skill 被选中或被实际调用时，才读取完整 SKILL.md 注入。
  3. meta_summary 输出 ~100 tokens，load_full 单个 skill ~300~600 tokens，对比可见省 window。
"""
import re
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# 只抓 frontmatter（首对 --- 之间），正文此时不读 —— 这就是「渐进」的落点
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
# 块标量指示符：description: >- / | / > 等，值在后续缩进行里
_BLOCK_SCALAR = {">", ">-", "|", "|-"}


def _parse_frontmatter(text: str) -> dict:
    """解析 frontmatter 为 dict；支持 YAML 块标量（>- / | 等，flash-card 用到）。"""
    fields: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(\w+):\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in _BLOCK_SCALAR:  # 块标量：收集后续缩进行
            block, i = [], i + 1
            while i < len(lines) and lines[i][:1] in (" ", "\t"):
                block.append(lines[i].strip())
                i += 1
            fields[key] = " ".join(block)
        else:
            fields[key] = val
            i += 1
    return fields


@dataclass
class SkillMeta:
    name: str  # 与目录名一致，是 Action 调用名（可含连字符）
    description: str  # 预加载阶段给 LLM 看的唯一信息
    path: Path  # skill 目录，load_full 时据此读 SKILL.md


class SkillRegistry:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._skills_dir = skills_dir
        self._metas: dict[str, SkillMeta] = {}
        self._full_cache: dict[str, str] = {}  # 完整内容只读一次，命中后走缓存
        self._scan()

    def _scan(self):
        """扫描 skills/，仅解析 frontmatter —— Phase 1"""
        if not self._skills_dir.exists():
            return
        for skill_dir in sorted(self._skills_dir.iterdir()):
            md = skill_dir / "SKILL.md"
            if not md.exists():
                continue
            m = _FRONTMATTER_RE.match(md.read_text(encoding="utf-8"))
            if not m:
                continue
            fields = _parse_frontmatter(m.group(1))
            name = fields.get("name", skill_dir.name)
            self._metas[name] = SkillMeta(
                name=name,
                description=self._clean_desc(fields.get("description", "")),
                path=skill_dir,
            )

    @staticmethod
    def _clean_desc(desc: str) -> str:
        # frontmatter 多行描述会被正则压成单行，去掉多余空白
        return " ".join(desc.split())

    # ── Phase 1：轻量 meta 列表 ──────────────────────────────────────
    def all_metas(self) -> list[SkillMeta]:
        return list(self._metas.values())

    def meta_summary(self) -> str:
        lines = [f"- {m.name}: {m.description}" for m in self._metas.values()]
        return "\n".join(lines) or "（无可用 skill）"

    # ── Phase 2 / 懒加载：完整 SKILL.md ──────────────────────────────
    def load_full(self, name: str) -> str:
        if name in self._full_cache:
            return self._full_cache[name]
        if name not in self._metas:
            return ""
        self._full_cache[name] = (self._metas[name].path / "SKILL.md").read_text(encoding="utf-8")
        return self._full_cache[name]

    def load_multiple(self, names: list[str]) -> str:
        """拼接多个 skill 完整说明，用于 Phase 2 一次性注入。"""
        parts = []
        for n in names:
            full = self.load_full(n)
            if full:
                parts.append(f"=== SKILL: {n} ===\n{full}")
        return "\n\n".join(parts)

    # ── 工具方法 ─────────────────────────────────────────────────────
    def exists(self, name: str) -> bool:
        return name in self._metas

    def script_dir(self, name: str) -> Path | None:
        if name not in self._metas:
            return None
        d = self._metas[name].path / "scripts"
        return d if d.exists() else None
