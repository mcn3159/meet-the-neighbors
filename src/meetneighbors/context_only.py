"""Center-only GLM embeddings with raw ESM masking before normalization."""

import csv
from dataclasses import dataclass
from importlib import resources
from itertools import islice
import logging
from pathlib import Path
import pickle

import numpy as np
import torch


ESM_DIM = 1280
MAX_PROTEINS = 30
LAYER = 9  # Hidden-state 0 is the embedding output, 9 is encoder block 9.
ESM_FILENAME = "all_glm_input_prots_reps.esm.embs.pkl"
OUTPUT_FILENAME = "context_only_layer9.tsv"


@dataclass(frozen=True)
class Neighborhood:
    source_file: str
    source_row: int
    name: str
    proteins: tuple[str, ...]
    center_index: int
    center_id: str


def iter_neighborhoods(paths, input_mode):
    """Read headerless TSVs, retaining physical (one-based) source row numbers."""
    if input_mode not in {"masked", "original"}:
        raise ValueError("input_mode must be masked or original")
    for path in paths:
        with path.open(newline="") as handle:
            for row_number, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
                location = f"{path}:{row_number}"
                if len(fields) != 3 or not fields[0]:
                    raise ValueError(f"{location}: expected neighborhood, proteins, center index")
                proteins = tuple(fields[1].split(";"))
                if not 1 <= len(proteins) <= MAX_PROTEINS:
                    raise ValueError(f"{location}: expected 1–{MAX_PROTEINS} proteins")
                if any(len(p) < 2 or p[0] not in "+-" or not p[1:].strip() for p in proteins):
                    raise ValueError(f"{location}: each protein needs a +/- strand prefix and an ID")
                try:
                    center = int(fields[2])
                except ValueError as exc:
                    raise ValueError(f"{location}: center index must be an integer") from exc
                if not 0 <= center < len(proteins):
                    raise ValueError(f"{location}: center index {center} is out of range")
                center_id = proteins[center][1:]
                if input_mode == "masked":
                    if not center_id.endswith("_masked") or center_id == "_masked":
                        raise ValueError(f"{location}: masked mode requires a center ID ending in _masked")
                    center_id = center_id.removesuffix("_masked")
                yield Neighborhood(path.name, row_number, fields[0], proteins, center, center_id)


def load_esm_embeddings(path):
    with path.open("rb") as handle:
        entries = pickle.load(handle)
    embeddings = {}
    for key, value in entries:
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path}: ESM IDs must be nonempty strings")
        # The existing GLM batcher uses the first whitespace-delimited ID.
        key = key.split()[0]
        vector = np.asarray(value, dtype=np.float16)
        if vector.shape != (ESM_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"{path}: {key}: expected {ESM_DIM} finite ESM features")
        if key in embeddings and not np.array_equal(embeddings[key], vector):
            raise ValueError(f"{path}: conflicting ESM vectors for {key}")
        embeddings[key] = vector
    return embeddings


def load_normalization():
    path = resources.files("meetneighbors.predictvfs.data.pkl_objs").joinpath("norm.pkl")
    with path.open("rb") as handle:
        factors = pickle.load(handle)
    mean, std = np.asarray(factors["mean"]), np.asarray(factors["std"])
    if (mean.shape != (ESM_DIM,) or std.shape != (ESM_DIM,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("norm.pkl must contain finite 1280-dimensional mean and positive std")
    return mean, std


def prepare_batch(rows, embeddings, mean, std):
    """Match batch_data's float16 staging, then promote to float32 for GLM.

    Mask by occurrence, never by replacing a shared protein's dictionary entry.
    Only the 1280 ESM features are masked; the strand remains +/-0.5.
    """
    inputs = np.zeros((len(rows), MAX_PROTEINS, ESM_DIM + 1), dtype=np.float16)
    attention = np.zeros((len(rows), MAX_PROTEINS), dtype=np.int64)
    raw_mask = np.full(ESM_DIM, -1, dtype=np.float16)
    for batch_index, row in enumerate(rows):
        for position, protein in enumerate(row.proteins):
            if position == row.center_index:
                raw = raw_mask
            else:
                try:
                    raw = embeddings[protein[1:]]
                except KeyError as exc:
                    raise ValueError(
                        f"{row.source_file}:{row.source_row}: missing neighbor ESM embedding {protein[1:]}"
                    ) from exc
            inputs[batch_index, position, :ESM_DIM] = (raw - mean) / std
            inputs[batch_index, position, ESM_DIM] = 0.5 if protein[0] == "+" else -0.5
            attention[batch_index, position] = 1
        if not np.isfinite(inputs[batch_index]).all():
            raise ValueError(f"{row.source_file}:{row.source_row}: normalized ESM features are not finite")
    return torch.from_numpy(inputs).float(), torch.from_numpy(attention)


def select_device(name):
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available; use --device cpu")
    return torch.device(name)


def load_model(device):
    from transformers import RobertaConfig
    from meetneighbors.predictvfs.glm.gLM import gLM
    from meetneighbors.predictvfs.loader import get_glm_model_path

    # Match neighborhood_classification.create_glm_embeds checkpoint architecture.
    config = RobertaConfig(
        vocab_size=30522, max_position_embedding=MAX_PROTEINS,
        hidden_size=ESM_DIM, num_attention_heads=10, type_vocab_size=1,
        tie_word_embeddings=False, num_hidden_layers=19, num_pc=100,
        num_pred=4, predict_probs=True, emb_dim=ESM_DIM + 1,
        output_attentions=False, output_hidden_states=True,
        position_embedding_type="relative_key_query",
    )
    model = gLM(config)
    state = torch.load(get_glm_model_path(), map_location=device)
    incompatible = model.load_state_dict(state, strict=False)
    # Older Transformers checkpoints can have additional position_ids buffers.
    # Never silently use randomly initialized weights for the embedding pathway.
    missing = [key for key in incompatible.missing_keys
               if key.startswith(("dense.", "roberta.roberta."))]
    if missing:
        raise ValueError(f"GLM checkpoint is missing embedding weights: {missing}")
    del state
    return model.eval().to(device)


def embed_centers(model, inputs, attention, rows, device):
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        # Bypass the prediction/loss heads: these do not contribute to hidden states.
        outputs = model.roberta.roberta(
            inputs_embeds=model.dense(inputs.to(device)),
            attention_mask=attention.to(device),
            output_attentions=False, output_hidden_states=True, return_dict=True,
        )
        hidden = outputs.hidden_states[LAYER]
        centers = torch.tensor([row.center_index for row in rows], device=device)
        vectors = hidden[torch.arange(len(rows), device=device), centers]
        vectors = vectors.float().cpu().numpy()
    if vectors.shape != (len(rows), ESM_DIM) or not np.isfinite(vectors).all():
        raise ValueError("GLM returned invalid layer-9 center embeddings")
    return vectors


def run_context_only(input_dir, out, batch_size=100, device="auto", input_mode="masked", logger=None):
    logger = logger or logging.getLogger(__name__)
    if batch_size < 1:
        raise ValueError("--glm_bs must be a positive integer")
    input_dir, out = Path(input_dir).expanduser(), Path(out).expanduser()
    output = out / OUTPUT_FILENAME
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    # Match the notebook glob: skip hidden files, including macOS ._ sidecars.
    paths = sorted(path for path in (input_dir / "glm_inputs").glob("*.tsv")
                   if path.is_file() and not path.name.startswith("."))
    if not paths:
        raise ValueError(f"No input TSVs found in {input_dir / 'glm_inputs'}")
    device = select_device(device)
    embeddings = load_esm_embeddings(input_dir / "glm_outputs" / ESM_FILENAME)
    mean, std = load_normalization()

    # Validate every row before loading the large model or creating a result file.
    count = 0
    for row in iter_neighborhoods(paths, input_mode):
        prepare_batch([row], embeddings, mean, std)
        count += 1
    if not count:
        raise ValueError("Input TSVs contain no neighborhoods")
    logger.info("Embedding %d centers from %d files on %s (layer %d)", count, len(paths), device, LAYER)
    model = load_model(device)
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing results. Remove our partial file on failure.
    with output.open("x", newline="") as handle:
        try:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["source_file", "source_row", "neighborhood_name", "rep", *range(ESM_DIM)])
            iterator = iter_neighborhoods(paths, input_mode)
            completed = 0
            while rows := list(islice(iterator, batch_size)):
                inputs, attention = prepare_batch(rows, embeddings, mean, std)
                vectors = embed_centers(model, inputs, attention, rows, device)
                for row, vector in zip(rows, vectors):
                    writer.writerow([row.source_file, row.source_row, row.name, row.center_id, *vector])
                completed += len(rows)
                logger.info("Embedded %d/%d centers", completed, count)
        except BaseException:
            output.unlink()
            raise
    logger.info("Saved center embeddings to %s", output)
    return output
