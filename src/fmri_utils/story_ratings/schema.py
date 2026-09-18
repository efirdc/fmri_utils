from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .config import RatingSpec

# Single-letter keys cost far fewer output tokens than full field names, which
# the model would otherwise repeat for every segment. They are expanded back
# before anything is stored, so only the request differs.
_RESERVED_COMPACT = {"i": "index"}


def compact_key_map(spec: RatingSpec) -> Dict[str, str]:
    """Map short request keys to the stored field names."""
    mapping = dict(_RESERVED_COMPACT)
    used = set(mapping)
    for name in spec.field_names:
        if name == "index":
            continue
        for candidate in (name[0], *(f"{name[0]}{i}" for i in range(1, 10))):
            if candidate not in used:
                mapping[candidate] = name
                used.add(candidate)
                break
        else:  # pragma: no cover - only with absurdly many fields
            raise ValueError(f"cannot assign a compact key for {name!r}")
    return mapping


def build_schema(spec: RatingSpec, *, compact: bool = False, bounded_integers: bool = True) -> Dict[str, Any]:
    """JSON schema for one batch of ratings.

    ``bounded_integers=False`` replaces ``minimum``/``maximum`` with an
    ``enum``, which the Anthropic structured-output validator requires.
    """
    spec.validate()
    inverse = {value: key for key, value in compact_key_map(spec).items()} if compact else {}
    name_of = lambda field: inverse.get(field, field) if compact else field  # noqa: E731

    def integer(minimum: int, maximum: int) -> Dict[str, Any]:
        if bounded_integers:
            return {"type": "integer", "minimum": minimum, "maximum": maximum}
        return {"type": "integer", "enum": list(range(minimum, maximum + 1))}

    properties: Dict[str, Any] = {name_of("index"): {"type": "integer"}}
    properties[name_of(spec.scale.name)] = integer(spec.scale.minimum, spec.scale.maximum)
    for flag in spec.flags:
        properties[name_of(flag)] = integer(0, 1)
    for categorical, choices in spec.categoricals.items():
        properties[name_of(categorical)] = {"type": "string", "enum": list(choices)}
    if spec.include_confidence:
        properties[name_of("confidence")] = integer(1, 3)
    if spec.include_reason:
        properties[name_of("reason")] = {"type": "string"}

    item = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"ratings": {"type": "array", "items": item}},
        "required": ["ratings"],
        "additionalProperties": False,
    }


def expand_compact(payload: Mapping[str, Any], spec: RatingSpec) -> Dict[str, Any]:
    """Rename single-letter keys back to the stored field names."""
    mapping = compact_key_map(spec)
    ratings = payload.get("ratings")
    if not isinstance(ratings, list):
        return dict(payload)
    return {"ratings": [{mapping.get(key, key): value for key, value in item.items()} for item in ratings]}


def validate_ratings(payload: Mapping[str, Any], n_items: int, spec: RatingSpec) -> List[Dict[str, Any]]:
    """Check that a response rates every item exactly once, and sort by index.

    Models sometimes drop items from a long list. A short list would silently
    misalign every later rating, so this raises instead.
    """
    ratings = payload.get("ratings")
    if not isinstance(ratings, list) or len(ratings) != n_items:
        got = len(ratings) if isinstance(ratings, list) else "none"
        raise ValueError(f"expected {n_items} ratings, got {got}")
    ordered = sorted(ratings, key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in ordered] != list(range(n_items)):
        raise ValueError("rating indices are not exactly 0..n-1")
    for item in ordered:
        value = int(item[spec.scale.name])
        if not spec.scale.minimum <= value <= spec.scale.maximum:
            raise ValueError(f"{spec.scale.name}={value} is outside the scale")
        for categorical, choices in spec.categoricals.items():
            if item[categorical] not in choices:
                raise ValueError(f"{categorical}={item[categorical]!r} is not one of {choices}")
    return ordered


def build_prompt(template: str, texts: Sequence[str], placeholder: str = "{utterances}") -> str:
    """Insert numbered segments into the rubric."""
    lines = "\n".join(f"{index:03d} | {text}" for index, text in enumerate(texts))
    return template.replace(placeholder, lines)
