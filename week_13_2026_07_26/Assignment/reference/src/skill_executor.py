"""
Skill 执行器 —— 把 ReAct 的 Action 文本路由到对应脚本并执行

教学重点：
  1. Action 解析：`skill-name(key=value, ...)` → (name, params_dict)
     skill 名允许连字符（flash-card）；参数用 ast.literal_eval，支持单/双引号、字典、列表字面量。
  2. 双形态执行：
     - 若脚本定义了 run(**kwargs) -> dict，直接调用（CLAUDE.md §12.2 的标准形态）；
     - 否则按 CLI 脚本执行：把 params 写成 JSON 丢进 skill 的 data/，再 `python <script> <json>`
       在 outputs/ 下运行，产物自然落到 outputs/。flash-card 的 make_flashcard.py 走这条。
  3. 统一返回 dict，便于序列化成 Observation 回喂给 LLM。
"""
import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from skill_registry import SkillRegistry

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# skill 名允许字母/数字/下划线/连字符（flash-card）
_NAME_RE = re.compile(r"^\s*([a-zA-Z][\w-]*)\s*\((.*)\)\s*$", re.DOTALL)


def parse_action(action_text: str) -> tuple[str, dict]:
    """解析 'flash-card(data={...})' → ('flash-card', {'data': {...}})

    同时支持关键字参数（name=...）与位置参数（"flash-card"）：
    位置参数收集到 params['_args'] 列表，便于 read_skill("xxx") 这种自然写法。
    """
    m = _NAME_RE.match(action_text.strip())
    if not m:
        return "", {}
    name, params_str = m.group(1), m.group(2).strip()
    params: dict = {}
    if params_str:
        try:
            tree = ast.parse(f"_f({params_str})", mode="eval")
            call = tree.body
            for arg in call.args:  # 位置参数 → _args 列表
                params.setdefault("_args", []).append(ast.literal_eval(arg))
            for kw in call.keywords:  # 关键字参数 → 各自键
                params[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            params["_raw"] = params_str  # 降级：整段当字符串
    return name, params


def _load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("_skill_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SkillExecutor:
    def __init__(self, registry: SkillRegistry):
        self._registry = registry

    def execute(self, action_text: str) -> dict:
        name, params = parse_action(action_text)
        if not name:
            return {"error": f"无法解析 Action: {action_text!r}"}
        if not self._registry.exists(name):
            return {"error": f"未知 skill: {name}"}
        sdir = self._registry.script_dir(name)
        if not sdir:
            return {"error": f"skill '{name}' 无 scripts 目录"}

        # 主脚本：与 skill 同名，否则取目录下唯一 .py
        script = sdir / f"{name}.py"
        if not script.exists():
            pys = list(sdir.glob("*.py"))
            if not pys:
                return {"error": f"{name} 的 scripts/ 下无 .py 文件"}
            script = pys[0]

        # 形态一：run(**kwargs) 标准入口
        try:
            mod = _load_module(script)
        except Exception as e:
            return {"error": f"加载脚本失败: {e}"}
        if hasattr(mod, "run"):
            try:
                return mod.run(**params)
            except TypeError as e:
                return {"error": f"参数错误: {e}"}
            except Exception as e:
                return {"error": f"执行异常: {e}"}

        # 形态二：CLI 脚本 —— params 落盘成 JSON，脚本拿文件路径当入参
        # 约定：LLM 把整份数据塞进 data= 参数；缺省则把 params 本身当数据
        data_content = params.get("data") if "data" in params else params
        if not isinstance(data_content, dict):
            data_content = {"value": data_content}
        # 用 word / name 字段做文件名，缺省用 skill 名
        key = (data_content.get("word") or data_content.get("name")
               or params.get("word") or name)
        scripts_dir = self._registry.script_dir(name)
        data_file = scripts_dir.parent / "data" / f"{key}.json"
        data_file.parent.mkdir(exist_ok=True)
        data_file.write_text(json.dumps(data_content, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        try:
            # cwd=outputs/ 让 make_flashcard.py 默认的 ./<word>.html 落到这里
            proc = subprocess.run(
                [sys.executable, str(script), str(data_file)],
                cwd=str(OUTPUTS_DIR),
                capture_output=True, text=True, timeout=60,
            )
        except Exception as e:
            return {"error": f"CLI 执行失败: {e}"}

        html_path = OUTPUTS_DIR / f"{key}.html"
        base = {"script": script.name, "data_file": str(data_file)}

        if proc.returncode != 0:
            # 脚本崩溃：只回吐 stderr 最后一行（真正的异常行，如 KeyError: 'word'）
            # 不把整段 Python traceback 灌给 LLM —— 噪音大且泄漏内部实现路径
            tail = (proc.stderr or "").strip().splitlines()
            last = tail[-1] if tail else f"退出码 {proc.returncode}（无 stderr）"
            base["error"] = f"脚本执行失败：{last}"
            return base

        # 成功：返回产物路径；stderr 若有非空告警也带上（便于排查但不是错误）
        base["stdout"] = proc.stdout.strip()
        if proc.stderr.strip():
            base["stderr"] = proc.stderr.strip()
        base["html_path"] = str(html_path) if html_path.exists() else None
        return base
