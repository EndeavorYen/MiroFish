"""Action taxonomy (taxonomy.yaml) -> System One decision trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.yaml")
EXIT_KEYS = ("none", "other")
MAX_OPTIONS = 26
NEEDS = {"post", "comment", "user", "query", "content"}


class TaxonomyError(ValueError):
    pass


@dataclass(frozen=True)
class Leaf:
    """What a decision path ends in: an OASIS action and what it needs."""

    path: tuple[str, ...]
    action: str
    needs: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Taxonomy:
    platform: str
    tree: dict[str, Any]  # ask_tree node: {name, question, children}
    leaves: dict[tuple[str, ...], Leaf]
    dialogue_kinds: dict[str, str]

    def leaf(self, path: list[str] | tuple[str, ...]) -> Leaf:
        return self.leaves[tuple(path)]

    @property
    def actions(self) -> set[str]:
        return {leaf.action for leaf in self.leaves.values()}


def _build(
    node: dict[str, Any], path: tuple[str, ...], leaves: dict[tuple[str, ...], Leaf]
) -> dict[str, Any]:
    options = node.get("options") or {}
    if not options:
        raise TaxonomyError(f"{'/'.join(path) or 'root'} has no options")
    if len(options) > MAX_OPTIONS:
        raise TaxonomyError(f"{'/'.join(path) or 'root'} has {len(options)} options (max 26)")
    if not any(key in options for key in EXIT_KEYS):
        raise TaxonomyError(f"{'/'.join(path) or 'root'} needs a none/other option")
    children: dict[str, Any] = {}
    criteria: dict[str, str] = {}
    for key, option in options.items():
        criteria[key] = option.get("description", key)
        child_path = path + (key,)
        if "options" in option:
            children[key] = _build(option, child_path, leaves)
        else:
            if "action" not in option:
                raise TaxonomyError(f"leaf {'/'.join(child_path)} has no action")
            needs = tuple(option.get("needs") or ())
            unknown = set(needs) - NEEDS
            if unknown:
                raise TaxonomyError(f"leaf {'/'.join(child_path)} has unknown needs {unknown}")
            leaves[child_path] = Leaf(child_path, option["action"], needs)
    tree = {
        "name": node.get("name") or (path[-1] if path else "action"),
        "question": {
            "type": "choice",
            "instructions": node.get("instructions", "選一個"),
            "criteria": criteria,
        },
    }
    if children:
        tree["children"] = children
    return tree


def parse_taxonomy(data: dict[str, Any], platform: str) -> Taxonomy:
    if platform not in data:
        raise TaxonomyError(f"no taxonomy for platform {platform!r}")
    leaves: dict[tuple[str, ...], Leaf] = {}
    tree = _build(data[platform], (), leaves)
    kinds = dict(data.get("dialogue_kinds") or {})
    if kinds and not any(key in kinds for key in EXIT_KEYS):
        raise TaxonomyError("dialogue_kinds needs an other/none option")
    if len(kinds) > MAX_OPTIONS:
        raise TaxonomyError("dialogue_kinds has more than 26 options")
    return Taxonomy(platform, tree, leaves, kinds)


@lru_cache(maxsize=None)
def load_taxonomy(platform: str, path: str = str(TAXONOMY_PATH)) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return parse_taxonomy(data, platform)
