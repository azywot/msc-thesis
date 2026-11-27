"""Single-pass placeholder rendering for overridable sub-agent prompt templates."""

import re
from typing import Dict, List

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def render_placeholders(template: str, values: Dict[str, str]) -> str:
    """Replace ``{name}`` for names in ``values``; leave all other text untouched.

    Rendering is a single pass, so substituted values are never re-scanned.
    """
    return _PLACEHOLDER.sub(
        lambda m: values[m.group(1)] if m.group(1) in values else m.group(0), template
    )


def find_placeholders(template: str) -> List[str]:
    """Return placeholder names in order of appearance, with repeats."""
    return _PLACEHOLDER.findall(template)
