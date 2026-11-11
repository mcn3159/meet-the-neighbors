"""Bounded GLM batches, occurrence-based restoration, and resumable TSV output."""

import csv
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from meetneighbors.glm_input_frm_neighbors import Neighborhood
from meetneighbors.neighborhood_order import ordered_occurrences, query_labels

METADATA = ['glminput_tsv', 'query', 'neighborhood_name', 'rep', 'nn_center_id']
DIMENSIONS = list(range(1280))
BATCH_FORMAT_VERSION = 2


def embedding_columns(frame):
    columns = [i if i in frame.columns else str(i) for i in DIMENSIONS]
    if not all(c in frame.columns for c in columns):
        raise ValueError('Expected GLM embedding dimensions 0 through 1279')
    return columns


def read_neighborhoods(input_dir, clusters):
    """Read the surviving TSVs, retaining their actual filenames as provenance."""
    groups = {name: grp for name, grp in clusters.groupby('neighborhood_name')}
    rows, seen = [], {}
    for path in sorted(Path(input_dir).glob('*.tsv')):
        if path.name.startswith('._'):
            continue
        with path.open(newline='') as handle:
            for line, fields in enumerate(csv.reader(handle, delimiter='\t'), 1):
                if len(fields) != 3:
                    raise ValueError(f'{path}:{line}: expected three GLM input fields')
                name, proteins, center = fields
                if name not in groups:
                    raise ValueError(f'{path}:{line}: neighborhood absent from mmseqs_clust.tsv: {name}')
                raw = groups[name]
                ordered = ordered_occurrences(raw)
                proteins, center = tuple(proteins.split(';')), int(center)
                expected = tuple(ordered.strand + ordered.rep)
                center_name = ordered.VF_center.iloc[0] + '!!!' + ordered.gff.iloc[0]
                indices = ordered.index[ordered.prot_gffname.eq(center_name)].tolist()
                if proteins != expected or center not in indices:
                    raise ValueError(f'{path}:{line}: protein order or center differs from mmseqs_clust.tsv')
                labels = tuple(query_labels(raw.loc[raw.prot_gffname.eq(center_name), 'query']))
                if not labels:
                    raise ValueError(f'{name}: VF center has no query labels')
                row = Neighborhood(str(path), name, labels, proteins, center, proteins[center][1:])
                identity = (proteins, center, labels)
                if name in seen:
                    if seen[name] != identity:
                        raise ValueError(f'Conflicting GLM inputs for {name}')
                    continue
                seen[name] = identity
                rows.append(row)
    if set(seen) != set(groups):
        raise ValueError(f'GLM inputs do not cover mmseqs_clust.tsv; missing {sorted(set(groups) - set(seen))[:5]}')
    if not rows:
        raise ValueError('No neighborhoods to embed')
    return rows


def hash_format(mapping):
    if not mapping:
        return 'none'
    lengths = {len(entry) for entries in mapping.values() for entry in entries}
    if lengths == {3}:
        return 'legacy'
    if lengths == {4}:
        return 'indexed'
    raise ValueError('Hash mapping must contain uniformly three-field or four-field entries')


def center_targets(rows, clusters, mapping):
    """Return (source row, position, metadata), never join vectors on rep alone."""
    mode = hash_format(mapping)
    if mode == 'none':
        groups = {name: group for name, group in clusters.groupby('neighborhood_name')}
        targets = []
        for i, row in enumerate(rows):
            raw = groups[row.neighborhood_name]
            ordered = ordered_occurrences(raw)
            if tuple(ordered.strand + ordered.rep) != row.proteins:
                raise ValueError(f'{row.neighborhood_name}: protein order differs from mmseqs_clust.tsv')
            center_name = ordered.VF_center.iloc[0] + '!!!' + ordered.gff.iloc[0]
            indices = ordered.index[ordered.prot_gffname.eq(center_name)]
            if row.center_index not in indices:
                raise ValueError(f'{row.neighborhood_name}: invalid reference center index')
            starts = pd.to_numeric(raw.start, errors='raise')
            for position in indices:
                occurrence = ordered.iloc[position]
                labels = query_labels(raw.loc[starts.eq(occurrence.start), 'query'])
                if not labels:
                    raise ValueError(f'{row.neighborhood_name}: center at index {position} has no query labels')
                targets.extend((i, int(position), query, occurrence.rep, row.neighborhood_name)
                               for query in labels)
    else:
        names = clusters[['neighborhood_name', 'nn_hashes']].drop_duplicates()
        if names.neighborhood_name.duplicated().any() or names.nn_hashes.isna().any():
            raise ValueError('Each retained neighborhood must map to exactly one hash')
        hashes = dict(zip(names.neighborhood_name, names.nn_hashes.astype(str)))
        normalized = {str(key): entries for key, entries in mapping.items()}
        if mode == 'legacy':
            from meetneighbors.compute_umap import legacy_center_targets
            targets = legacy_center_targets(rows, hashes, normalized)
        else:
            targets = []
            for i, row in enumerate(rows):
                key = hashes[row.neighborhood_name]
                if key not in normalized:
                    raise ValueError(f'Missing hash mapping for {row.neighborhood_name}')
                for query, rep, name, position in normalized[key]:
                    if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
                        raise ValueError(f'{name}: center index must be an integer')
                    if not 0 <= position < len(row.proteins) or row.proteins[position][1:] != rep:
                        raise ValueError(f'{name}: center index {position} does not match representative {rep}')
                    for label in query_labels([query]):
                        targets.append((i, int(position), label, rep, name))
    result, seen = [], set()
    for i, position, query, rep, name in targets:
        metadata = (Path(rows[i].source_file).stem, query, name, rep, position)
        key = (query, name, rep, position)
        if key in seen:
            raise ValueError(f'Duplicate restored VF center: {key}')
        seen.add(key)
        result.append((i, position, metadata))
    return result


def center_dataframe(rows, vectors, targets):
    if vectors.ndim != 3 or vectors.shape[0] != len(rows) or vectors.shape[2] != 1280:
        raise ValueError('Expected GLM vectors shaped (neighborhoods, padded proteins, 1280)')
    if vectors.shape[1] < max(len(row.proteins) for row in rows):
        raise ValueError('GLM output is shorter than its input neighborhoods')
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite GLM embeddings')
    values = np.asarray([vectors[i, position] for i, position, _ in targets], dtype=np.float32)
    return pd.concat([pd.DataFrame([m for _, _, m in targets], columns=METADATA),
                      pd.DataFrame(values.reshape(-1, 1280), columns=DIMENSIONS)], axis=1)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


def run_embedding_batches(rows, clusters, mapping, output_dir, embed_batch, batch_size,
                          max_proteins, resume=False, identity=None, logger=None):
    """Persist only restored centers; the callback owns lazy model initialization."""
    if batch_size < 1:
        raise ValueError('--glm_bs must be positive')
    if any(not 0 < len(row.proteins) <= max_proteins for row in rows):
        raise ValueError('Neighborhood length exceeds --max_prots or is empty')
    logger = logger or logging.getLogger(__name__)
    mode = hash_format(mapping)
    if mode == 'legacy':
        logger.warning('Legacy hash mapping without center indices: using compute_umap.py legacy occurrence selection')
    all_targets = center_targets(rows, clusters, mapping)
    if not all_targets:
        raise ValueError('No labeled VF centers to save')
    if mapping:
        if mode == 'indexed':
            expected = {(q, e[2], e[1], e[3]) for entries in mapping.values()
                        for e in entries for q in query_labels([e[0]])}
            actual = {(m[1], m[2], m[3], m[4]) for _, _, m in all_targets}
        else:
            expected = {(q, e[2]) for entries in mapping.values() for e in entries for q in query_labels([e[0]])}
            actual = {(m[1], m[2]) for _, _, m in all_targets}
        if actual != expected:
            raise ValueError('Restored centers do not cover the hash mapping')
    targets_by_row = {}
    for target in all_targets:
        targets_by_row.setdefault(target[0], []).append(target)
    spec = {'version': BATCH_FORMAT_VERSION, 'layer': 3, 'batch_size': batch_size, 'max_proteins': max_proteins,
            'rows': [asdict(row) for row in rows], 'targets': all_targets, 'dependencies': identity}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'embedding_batches.json'
    manifest = {'version': BATCH_FORMAT_VERSION, 'fingerprint': fingerprint, 'completed': {},
                'batches': [f'batch_{i:06d}.glm_embeds.tsv' for i in
                            range((len(rows) + batch_size - 1) // batch_size)]}
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('version', 1) != BATCH_FORMAT_VERSION:
            raise ValueError('Cached GLM batches predate duplicate-center support; rerun without --resume')
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('Embedding inputs, layer, or batch configuration changed; rerun without --resume')
    _atomic_json(manifest_path, manifest)
    paths = []
    for number, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start:start + batch_size]
        targets = [(i - start, position, metadata)
                   for i in range(start, start + len(batch))
                   for _, position, metadata in targets_by_row.get(i, [])]
        path = output_dir / f'batch_{number:06d}.glm_embeds.tsv'
        saved = manifest['completed'].get(path.name)
        if not (resume and saved and path.is_file() and file_digest(path) == saved):
            logger.info('Computing GLM batch %d (%d neighborhoods)', number, len(batch))
            vectors = embed_batch(batch)
            frame = center_dataframe(batch, vectors, targets)
            temporary = path.with_suffix('.tsv.tmp')
            frame.to_csv(temporary, sep='\t', index=False)
            temporary.replace(path)
            manifest['completed'][path.name] = file_digest(path)
            _atomic_json(manifest_path, manifest)
            del vectors, frame
        paths.append(path)
    combined = output_dir.parent / 'glm_embeds.tsv'
    temporary = combined.with_suffix('.tsv.tmp')
    with temporary.open('w') as destination:
        for i, path in enumerate(paths):
            with path.open() as source:
                if i:
                    next(source)
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(combined)
    return combined


def saved_batch_paths(output_dir):
    """Only expose complete, verified batches belonging to the current run."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / 'embedding_batches.json'
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    result = []
    for name in manifest['batches']:
        path = output_dir / name
        if (name not in manifest['completed'] or not path.is_file() or
                file_digest(path) != manifest['completed'][name]):
            raise ValueError('GLM batches are incomplete or corrupt; resume extraction first')
        result.append(path)
    return result
