"""Per-dataset facts that are not the loader's business.

Before this module these lived as string literals in three places:
``prompts/builder.py``'s template dispatch, ``runner/metrics.py``'s
``_STRATIFIED`` set, and ``runner/metrics.py``'s ``level_key``.  Adding a
benchmark meant finding all three.  Now it means adding one row here.

Keyed by dataset **name**, deliberately not by ``DatasetRegistry`` membership:
``"math"`` and ``"deepmath"`` have templates but no loader (they are RL
data-prep names) and must keep resolving to the math template.

Two subtleties preserved from the code this replaced, both easy to "clean up"
into a behaviour change:

*Lookup is case-sensitive.*  ``level_key`` and ``_STRATIFIED`` compared the raw
``dataset_name``, so ``"GAIA"`` was *not* stratified and had no level field,
while ``builder.py`` lowercased before dispatching.  ``get_spec`` therefore does
not fold case; the builder lowercases at its call site, as it always did.

*``template=None`` means "no mapping".*  The old dispatch left
``template_name = dataset_name`` for unrecognised names, so the lookup failed
and the ``FileNotFoundError`` handler logged a warning before falling back to
the base template.  Returning ``"base"`` here instead would silently skip that
warning, which is a run artefact in ``experiment.log``.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetSpec:
    """How one dataset is prompted and scored.

    Attributes:
        template: Prompt-template stem, or ``None`` when the dataset has no
            mapping and the raw name should be tried (then fall back to base).
        stratified: Whether ``compute_metrics`` emits a ``per_level`` breakdown.
        level_field: ``example.metadata`` key holding the stratification level.
        level_fallback_field: Key consulted when *level_field* is **absent**.
            Note "absent", not "None": the original used ``dict.get(k, default)``,
            so a key present with value ``None`` yields ``"None"``, not the
            fallback.
    """

    template: Optional[str] = None
    stratified: bool = False
    level_field: Optional[str] = None
    level_fallback_field: Optional[str] = None


_DEFAULT = DatasetSpec()

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "gaia":         DatasetSpec("gaia", True, "level"),
    "hle":          DatasetSpec("gaia", True, "category"),
    "musique":      DatasetSpec("gaia", False),
    "aime":         DatasetSpec("math", True, "year"),
    "math500":      DatasetSpec("math", True, "difficulty", "year"),
    "amc":          DatasetSpec("math", True, "difficulty", "year"),
    "deepmath":     DatasetSpec("math", False),
    "math":         DatasetSpec("math", False),
    "gpqa":         DatasetSpec("gpqa", False),
    "bigcodebench": DatasetSpec("bigcodebench", False),
}


def get_spec(name: str) -> DatasetSpec:
    """Spec for *name*, or a safe default for unknown names.

    Case-sensitive on purpose -- see the module docstring.
    """
    return DATASET_SPECS.get(name or "", _DEFAULT)
