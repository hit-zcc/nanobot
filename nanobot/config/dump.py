"""打印 nanobot 真正启动时用的那份配置。

为什么需要它
------------
配置类的 bug 有个固定形状：改了某个值，行为没变，然后所有人跑去查行为。
但根因几乎从来不是那个值本身，而是**进程读到的根本是另一个值**——
键挂错了父节点、写成 camelCase 而代码认 snake_case、
某个 agent 的覆盖压过了 defaults，或者干脆有第二份配置文件在别的路径。

这些从两边都看不见：配置文件只能告诉你「写了什么」，日志只能告诉你「发生了什么」。
夹在中间那层——合并完、校验完、填好默认值的那棵树——从来没被打印过，
于是「我改的那个值，是不是你正在用的那个值」这个最该问的问题，没有便宜的答案。

所以：把生效的树打印出来，并给每个叶子标注它来自**文件**还是**默认值**。
一个明明设置过、却显示 default 的键，本身就是完整的 bug 现场。

输出时会脱敏，目的是让这份内容可以直接贴进聊天里问人「为什么我这个设置不生效」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanobot.config.schema import Config

# 按键名做大小写无关的子串匹配。故意放宽：新加一个 provider 的凭据字段时，
# 当天就该被盖住，而不依赖谁记得回来更新这份名单。
_SECRET_HINTS = (
    "key", "secret", "token", "password", "passwd",
    "credential", "auth", "signature",
)


def _is_secret(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def redact(key: str, value: Any) -> Any:
    """给密钥打码，但留下足以辨认「是哪一个」的头尾。"""
    if not isinstance(value, str) or not value:
        return value
    if not _is_secret(key):
        return value
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-2:]} ({len(value)} chars)"


class _Missing:
    """区分「文件里没有这个键」和「有这个键但值是 null」。"""

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return "<missing>"


_MISSING = _Missing()


def _walk(effective: Any, raw: Any, key: str = "") -> Any:
    """把每个叶子和它的来源配成对：来自文件，还是来自默认值。"""
    if isinstance(effective, dict):
        out: dict[str, Any] = {}
        raw_dict = raw if isinstance(raw, dict) else {}
        for k, v in effective.items():
            out[k] = _walk(v, raw_dict.get(k, _MISSING), k)
        return out
    if isinstance(effective, list):
        raw_list = raw if isinstance(raw, list) else []
        return [
            _walk(v, raw_list[i] if i < len(raw_list) else _MISSING, key)
            for i, v in enumerate(effective)
        ]
    return {
        "value": redact(key, effective),
        "from": "default" if raw is _MISSING else "file",
    }


def _load_raw(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # 文件读不出来在这里不算错误：那样所有值都会显示成 default，
        # 而这恰恰就是「为什么我的设置被忽略了」的答案。
        return {}


def build_dump(config: Config, config_path: Path) -> dict[str, Any]:
    """返回生效配置，且每个叶子都标注了来源。"""
    effective = config.model_dump(mode="json", by_alias=True)
    raw = _load_raw(config_path)
    return {
        "configPath": str(config_path),
        "configExists": config_path.exists(),
        "workspace": str(config.workspace_path),
        "settings": _walk(effective, raw),
    }


def render_dump(dump: dict[str, Any], *, show_defaults: bool = True) -> str:
    """把带来源标注的树渲染成缩进文本，文件里设过的值打点标记。"""
    lines = [
        f"config: {dump['configPath']}" + ("" if dump["configExists"] else "  (does not exist)"),
        f"workspace: {dump['workspace']}",
        "",
    ]

    def emit(node: Any, name: str, depth: int) -> None:
        # 顶层用 depth=-1 调用，所以这里 +1 让它落在零缩进上。
        pad = "  " * (depth + 1)
        if isinstance(node, dict) and "value" in node and "from" in node:
            if node["from"] == "default" and not show_defaults:
                return
            mark = "•" if node["from"] == "file" else " "
            lines.append(f"{pad}{mark} {name}: {json.dumps(node['value'], ensure_ascii=False)}")
            return
        if isinstance(node, dict):
            start = len(lines)
            lines.append("")  # 占位；子树全被过滤掉时删掉，免得留一个空标题
            for k, v in node.items():
                emit(v, k, depth + 1)
            if lines[start + 1:]:
                lines[start] = f"{pad}{name}:"
            else:
                del lines[start]
            return
        if isinstance(node, list):
            start = len(lines)
            lines.append("")
            for i, item in enumerate(node):
                emit(item, f"[{i}]", depth + 1)
            if lines[start + 1:]:
                lines[start] = f"{pad}{name}:"
            else:
                del lines[start]
            return

    for key, value in dump["settings"].items():
        emit(value, key, -1)

    lines.append("")
    lines.append("• = 配置文件里显式设置的；其余都是内置默认值。")
    if not show_defaults:
        lines.append("(已隐藏默认值；去掉 --changed-only 可看全部)")
    return "\n".join(lines)
