"""The simulation API package resolves imports and paths from its new depth (#68).

The split moved the code from app/api/simulation.py to
app/api/simulation/<module>.py: function-level ``from ..x`` imports and
``__file__``-relative paths each needed one more level. These run the real
code paths, without monkeypatching them away.
"""

import os
import re
from pathlib import Path

from app import create_app
from app.api import simulation as simulation_api

PACKAGE = Path(simulation_api.__file__).parent
BACKEND = PACKAGE.parents[2]


def test_function_level_imports_resolve():
    ready, info = simulation_api._shared._check_simulation_prepared("no-such-simulation")
    assert ready is False and isinstance(info, dict)
    client = create_app().test_client()
    response = client.post("/api/simulation/prepare/status", json={"task_id": "no-such-task"})
    assert response.status_code == 404 and response.get_json()["success"] is False


def test_the_script_download_finds_the_scripts():
    client = create_app().test_client()
    response = client.get("/api/simulation/script/run_parallel_simulation.py/download")
    assert response.status_code == 200


def test_file_relative_paths_land_in_the_backend_directory():
    for module in PACKAGE.glob("*.py"):
        for literal in re.findall(r"""['"](\.\./[^'"{]*)""", module.read_text(encoding="utf-8")):
            resolved = Path(os.path.abspath(os.path.join(module.parent, literal)))
            assert resolved.parent == BACKEND or BACKEND in resolved.parents, (module.name, literal)
            assert (BACKEND / "app") not in [resolved, *resolved.parents], (module.name, literal)
