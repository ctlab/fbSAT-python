from ._abc import Task
from ._basic import BasicAutomatonTask
from ._basic_minimal import MinimalBasicAutomatonTask
from ._complete import CompleteAutomatonTask
from ._complete_minimal import MinimalCompleteAutomatonTask
from ._minimized import MinimizeAllGuardsTask

# Note: respect the order (Task -> Basic -> MinimalBasic -> Complete -> MinimalComplete)
