"""Simulation API routes (#68: split by resource from one 2,900-line module).

Each submodule registers its routes on ``simulation_bp``; the names are
re-exported so ``app.api.simulation.<name>`` keeps working. Patch a name a
route reads (``check_model_service``, ``SimulationManager``) in the module
that uses it, e.g. ``app.api.simulation._shared``.
"""

from . import _shared, entities, interview, monitor, prepare, records, run  # noqa: F401
from ._shared import *  # noqa: F401,F403
from ._shared import (  # noqa: F401
    _get_default_platform,
    INTERVIEW_PROMPT_PREFIX,
    optimize_interview_prompt,
    _check_simulation_prepared,
    _get_report_id_for_simulation,
    _model_service_problem,
)
from .entities import *  # noqa: F401,F403
from .interview import *  # noqa: F401,F403
from .monitor import *  # noqa: F401,F403
from .prepare import *  # noqa: F401,F403
from .records import *  # noqa: F401,F403
from .run import *  # noqa: F401,F403
