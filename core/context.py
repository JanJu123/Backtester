from dataclasses import dataclass
from typing import Optional

@dataclass
class SignalContext:
    """A container for strategy-specific information for a single trade signal."""
    fvg_level: Optional[float] = None
    # Add other context fields here in the future