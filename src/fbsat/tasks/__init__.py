from ._abc import Task
from ._full import FullAutomatonTask
from ._full_minimal import MinimalFullAutomatonTask
from ._partial import PartialAutomatonTask
from ._partial_minimal import MinimalPartialAutomatonTask
from ._complete import CompleteAutomatonTask
from ._complete_minimal import MinimalCompleteAutomatonTask
from ._minimized import MinimizeAllGuardsTask

# Note: respect the order (Task -> Partial -> MinimalPartial -> Complete -> MinimalComplete)
