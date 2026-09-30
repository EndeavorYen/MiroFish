"""Graph API: role confirmation before preparation (#64; split from graph.py in #68)."""

from flask import request, jsonify

from . import graph_bp
from ..graph.store import GraphNotFoundError
from ..models.project import ProjectManager
from ..services.zep_graph_memory_updater import ZepGraphMemoryManager
from ..utils.locale import t
from ..utils.logger import get_logger
from ..utils.zep_lifecycle import graph_lifecycle_lock
from .graph import _project_has_active_build

logger = get_logger('mirofish.api')


# ============== 角色確認（#64） ==============
#
# After the graph is built and before preparation, the user checks the roles
# the extractor found: rename a wrong boundary, merge duplicates, exclude
# roles that should not become agents, add ones it missed. Preparation reads
# the graph's typed entity nodes, so these edits change the prepared agents.
# Local graph backend only.

_ROLE_OPS = ("rename", "merge", "exclude", "include", "add")


def _local_role_store():
    from ..graph.local_store import LocalGraphStore
    from ..graph.store import get_graph_store

    store = get_graph_store()
    return store if isinstance(store, LocalGraphStore) else None


def _roles_payload(store, graph_id: str) -> dict:
    from ..graph.local_store import ROLE_EXCLUDED

    ontology = store.get_ontology(graph_id) or {}
    roles = []
    for node in store.list_nodes(graph_id):
        if "Entity" not in node.labels:
            continue
        types = [label for label in node.labels if label not in ("Entity", "Node")]
        roles.append({
            "uuid": node.uuid,
            "name": node.name,
            "type": types[0] if types else None,
            "excluded": bool(node.attributes.get(ROLE_EXCLUDED)),
            "aliases": node.attributes.get("aliases", []),
            "summary": node.summary,
        })
    return {
        "graph_id": graph_id,
        "entity_types": [t.get("name") for t in ontology.get("entity_types", [])],
        "roles": roles,
    }


@graph_bp.route('/<graph_id>/roles', methods=['GET'])
def list_roles(graph_id: str):
    """列出圖譜找到的角色（實體節點），供使用者在準備前確認。"""

    store = _local_role_store()
    if store is None:
        return jsonify({"success": False, "error": "role editing needs GRAPH_BACKEND=local"}), 501
    try:
        return jsonify({"success": True, "data": _roles_payload(store, graph_id)})
    except GraphNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404


@graph_bp.route('/<graph_id>/roles', methods=['POST'])
def edit_roles(graph_id: str):
    """依序套用角色修改：

    {"ops": [
        {"op": "rename", "uuid": "...", "name": "林書瑤"},
        {"op": "merge", "keep": "...", "drop": "..."},
        {"op": "exclude", "uuid": "..."},
        {"op": "include", "uuid": "...", "type": "Company"},
        {"op": "add", "name": "OpenAI", "type": "Company", "summary": "..."}
    ]}

    A failing op stops the list; the ops before it stay applied, and the
    response says which one failed.
    """

    store = _local_role_store()
    if store is None:
        return jsonify({"success": False, "error": "role editing needs GRAPH_BACKEND=local"}), 501
    ops = (request.get_json(silent=True) or {}).get("ops")
    if not isinstance(ops, list) or not ops:
        return jsonify({"success": False, "error": "ops must be a non-empty list"}), 400

    with graph_lifecycle_lock(graph_id):
        if any(_project_has_active_build(p) for p in ProjectManager.find_projects_by_graph_id(graph_id)):
            return jsonify({"success": False, "error": t('api.graphBuilding')}), 409
        # A simulation that still owns an updater (running, or failed with
        # writes to retry) writes agents back into the graph by name. Read
        # only: _active_graph_consumers would discard a failed run's updater.
        running = ZepGraphMemoryManager.get_simulation_ids_for_graph(graph_id)
        if running:
            return jsonify({"success": False, "error": f"simulations are using this graph: {', '.join(running)}"}), 409
        for index, op in enumerate(ops):
            kind = op.get("op") if isinstance(op, dict) else None
            try:
                # L3: field types first, so a bad value is a 400, not a 500.
                for field in ("uuid", "keep", "drop", "name", "type", "summary"):
                    if field in op and not isinstance(op[field], str):
                        raise ValueError(f"{field} must be a string")
                if kind == "rename":
                    store.rename_node(graph_id, op["uuid"], op["name"])
                elif kind == "merge":
                    store.merge_nodes(graph_id, op["keep"], op["drop"])
                elif kind == "exclude":
                    store.set_node_role(graph_id, op["uuid"], None)
                elif kind == "include":
                    store.set_node_role(graph_id, op["uuid"], op["type"])
                elif kind == "add":
                    store.add_role_node(graph_id, op["name"], op["type"], op.get("summary", ""))
                else:
                    raise ValueError(f"op must be one of {', '.join(_ROLE_OPS)}")
            except GraphNotFoundError as e:
                return jsonify({"success": False, "error": str(e), "failed_op": index}), 404
            except (KeyError, TypeError) as e:
                return jsonify({"success": False, "error": f"missing field {e}", "failed_op": index}), 400
            except ValueError as e:
                return jsonify({"success": False, "error": str(e), "failed_op": index}), 400
        logger.info("roles edited: graph=%s ops=%s", graph_id, [o.get("op") for o in ops])
        return jsonify({"success": True, "data": _roles_payload(store, graph_id)})
