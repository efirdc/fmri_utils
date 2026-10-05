"""Running the model over units and getting one vector each.

Heavy imports are deferred so that ``import fmri_utils.features`` costs
nothing: reading a cache, or resampling what is in it, should not need torch
installed. Only an actual extraction does.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from .pooling import pool_slots, pool_unit
from .spec import FeatureSpec, ModelSpec
from .units import Payload, Unit, build_payloads, build_stack_payloads


def select_device(requested: str = "") -> str:
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), max(size, 1)):
        yield items[start:start + size]


def load_transformer(spec: ModelSpec, device: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.huggingface_id, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModel.from_pretrained(spec.huggingface_id)
    vocab = getattr(model.config, "vocab_size", None)
    if vocab is not None and len(tokenizer) > int(vocab):
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    # Truncate from the left: when a context string is too long it is the
    # oldest context that should go, never the unit being embedded.
    tokenizer.truncation_side = "left"
    return tokenizer, model


def hidden_for_layer(outputs, layer: int | None):
    states = getattr(outputs, "hidden_states", None)
    if not states:
        raise ValueError("model returned no hidden_states; pass output_hidden_states=True")
    if layer is None:
        return states[-1]
    if layer >= len(states):
        raise ValueError(f"asked for block {layer}, model has {len(states) - 1}")
    return states[layer]


def load_whole_text_encoder(spec: ModelSpec, device: str) -> Callable[[list[str], int], np.ndarray]:
    """A model that returns one vector per input string, with no token access.

    Sentence encoders, CLIP text towers and LLM2Vec all work this way, so the
    unit's vector is the embedding of its whole context string. That is a
    coarser thing than the pooled variants and worth remembering when
    comparing: the unit is not isolated within its context.
    """
    if spec.family == "sentence_embedding":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(spec.huggingface_id, device=device)

        def encode(texts: list[str], batch_size: int) -> np.ndarray:
            prepared = [f"passage: {t}" if t.strip() else "passage: [empty]" for t in texts]
            return model.encode(prepared, batch_size=batch_size,
                                convert_to_numpy=True).astype(np.float32)
        return encode

    if spec.family == "clip_text":
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        tokenizer = AutoTokenizer.from_pretrained(spec.huggingface_id)
        model = CLIPTextModelWithProjection.from_pretrained(spec.huggingface_id)
        model.to(device).eval()

        def encode(texts: list[str], batch_size: int) -> np.ndarray:
            out = []
            with torch.inference_mode():
                for batch in chunked(texts, batch_size):
                    safe = [t if t.strip() else "[empty]" for t in batch]
                    encoded = tokenizer(safe, padding=True, truncation=True,
                                        return_tensors="pt").to(device)
                    features = model(**encoded).text_embeds
                    features = torch.nn.functional.normalize(features, dim=-1)
                    out.append(features.cpu().numpy().astype(np.float32))
            return np.concatenate(out, axis=0)
        return encode

    if spec.family == "llm2vec":
        import torch
        from huggingface_hub import snapshot_download
        from llm2vec import LLM2Vec

        # Resolved to local cache directories: compute nodes are offline, and
        # passing repo ids makes peft attempt a network lookup that fails.
        base = snapshot_download(spec.base_id or spec.huggingface_id, local_files_only=True)
        adapter = snapshot_download(spec.huggingface_id, local_files_only=True)
        model = LLM2Vec.from_pretrained(
            base, peft_model_name_or_path=adapter,
            device_map=device if device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="eager", pooling_mode="mean",
            max_length=spec.max_length,
        )

        def encode(texts: list[str], batch_size: int) -> np.ndarray:
            safe = [t if t.strip() else "[empty]" for t in texts]
            out = model.encode(safe, batch_size=batch_size)
            out = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
            return out.astype(np.float32)
        return encode

    raise ValueError(f"no whole-text encoder for family {spec.family!r}")


def embed_units(units: Sequence[Unit], spec: FeatureSpec, device: str = "",
                progress: Callable[[int, int], None] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """One vector per unit, plus a flag for whether any token backed it.

    Returns ``(embeddings, valid)``. Invalid units are zeros: a unit can be
    truncated out of its own context string, and the caller needs to be able to
    tell that from a genuine zero vector.
    """
    device = select_device(device)
    model_spec = spec.model
    stacked = model_spec.pooling == "context_slot_stack"
    payloads = (build_stack_payloads(units, spec.context.slots, spec.context.previous)
                if stacked else build_payloads(units, spec.context.previous))

    if model_spec.family != "transformer":
        encode = load_whole_text_encoder(model_spec, device)
        vectors = encode([p.context_text for p in payloads], model_spec.batch_size)
        valid = np.array([bool(p.unit_text.strip()) for p in payloads], dtype=bool)
        return vectors.astype(np.float32), valid

    import torch

    tokenizer, model = load_transformer(model_spec, device)
    vectors: list[np.ndarray] = []
    valid: list[bool] = []
    done = 0
    with torch.inference_mode():
        for batch in chunked(payloads, model_spec.batch_size):
            encoded = tokenizer(
                [p.context_text or "[empty]" for p in batch],
                padding=True, truncation=True, max_length=model_spec.max_length,
                return_tensors="pt", return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            offsets = encoded["offset_mapping"].numpy()
            special = encoded["special_tokens_mask"].numpy().astype(bool)
            attention = encoded["attention_mask"].numpy().astype(bool)
            inputs = {k: v.to(device) for k, v in encoded.items()
                      if k not in ("offset_mapping", "special_tokens_mask")}
            hidden = hidden_for_layer(
                model(**inputs, output_hidden_states=True), model_spec.layer
            ).cpu().numpy()

            for row, payload in enumerate(batch):
                usable = attention[row] & ~special[row]
                if stacked:
                    vector, backed = pool_slots(hidden[row], offsets[row], usable,
                                                payload.slot_spans)
                else:
                    vector, backed = pool_unit(hidden[row], offsets[row], usable,
                                               payload.start, payload.end,
                                               model_spec.pooling,
                                               payload.word_spans)
                vectors.append(vector)
                valid.append(backed)
            done += len(batch)
            if progress:
                progress(done, len(payloads))

    return np.stack(vectors).astype(np.float32), np.array(valid, dtype=bool)
