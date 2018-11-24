from ._abc import Task
from ._full import FullAutomatonTask
from ._full_minimal import MinimalFullAutomatonTask
from ._basic import BasicAutomatonTask
from ._basic_min import MinimalBasicAutomatonTask
from ._extended import ExtendedAutomatonTask
from ._extended_min import MinimalExtendedAutomatonTask
from ._extended_min_ub import MinimalExtendedUBAutomatonTask
from ._minimize import MinimizeGuardTask
from ._minimize_all import MinimizeAllGuardsTask

# Note: respect the order (Task -> FullAutomaton -> MinimalFullAutomaton)
# Note: respect the order (Task -> Basic -> MinimalBasic -> Extended -> MinimalExtended -> MinimalExtendedUB)
# Note: respect the order (Task -> MinimizeGuard -> MinimizeAllGuards)
