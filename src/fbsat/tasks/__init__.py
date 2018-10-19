from ._abc import Task
from ._full import FullAutomatonTask
from ._full_minimal import MinimalFullAutomatonTask
from ._partial import PartialAutomatonTask
from ._partial_minimal import MinimalPartialAutomatonTask
from ._complete import CompleteAutomatonTask
from ._complete_minimal import MinimalCompleteAutomatonTask
from ._complete_minimal_ub import MinimalCompleteUBAutomatonTask
from ._minimize import MinimizeGuardTask
from ._minimize_all import MinimizeAllGuardsTask

# Note: respect the order (Task -> FullAutomaton -> MinimalFullAutomaton)
# Note: respect the order (Task -> Partial -> MinimalPartial -> Complete -> MinimalComplete)
# Note: respect the order (Task -> MinimalGuard -> MinimizeAllGuards)
