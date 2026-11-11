"""Visualize saved genomic neighborhoods without loading the ML/extraction stack.

The public :func:`viz_neighborhoods` writes a portable, offline HTML document.
Coordinates and products always come from the originating GFF, never a cluster
representative. Only saved extraction membership is displayed.
"""

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import random
import shutil
import subprocess
from urllib.parse import unquote

from meetneighbors._neighborhood_html import HTML

LOG = logging.getLogger(__name__)


def _protein_id(value):
    return value.split("!!!")[0].split("|")[-1].split(":")[-1]


def _name_parts(name):
    parts = name.split("!!!")
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"Invalid neighborhood_name {name!r}; expected center!!!genome!!!contig!!!range")
    return parts


@dataclass(frozen=True)
class Occurrence:
    neighborhood: str
    protein: str
    start: int
    strand: str

    @property
    def key(self):
        _, genome, contig, _ = _name_parts(self.neighborhood)
        return (genome, contig, self.protein, self.start, self.strand)


def _occurrence(name, protein, start, strand):
    _name_parts(name)
    try:
        start = int(start)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid gene start {start!r} in {name}") from exc
    if start < 1 or strand not in ("+", "-") or not protein:
        raise ValueError(f"Invalid gene occurrence in {name}: {protein}, {start}, {strand}")
    return Occurrence(name, _protein_id(protein), start, strand)


def _encoded_occurrence(identifier):
    parts = identifier.split("!!!")
    if len(parts) != 7:
        raise ValueError(f"Invalid extracted FASTA identifier: {identifier!r}")
    return _occurrence("!!!".join(parts[1:5]), parts[0], parts[5], parts[6])


def _fasta(path):
    """Stream FASTA records, retaining only one sequence at a time."""
    name, sequence = None, []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence)
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Empty FASTA identifier in {path}")
                name, sequence = fields[0], []
            elif name is None:
                raise ValueError(f"Sequence before FASTA header in {path}")
            else:
                sequence.append(line)
    if name is not None:
        yield name, "".join(sequence)


def _rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        for number, row in enumerate(reader, 2):
            if any(row.get(col) is None for col in required):
                raise ValueError(f"{path}:{number}: incomplete row")
            yield row


def _put_assignment(mapping, key, rep, *, source=None):
    if key in mapping and mapping[key] != rep:
        location = f"{source}: " if source else ""
        raise ValueError(f"{location}Conflicting cluster assignments for {key}: {mapping[key]!r}, {rep!r}")
    mapping[key] = rep


@dataclass(frozen=True)
class _SavedClusterConflict:
    message: str


def _record_saved_assignment(mapping, key, rep, *, source):
    """Defer saved-data conflicts until we know which neighborhoods are used."""
    if isinstance(mapping.get(key), _SavedClusterConflict):
        return
    try:
        _put_assignment(mapping, key, rep, source=source)
    except ValueError as exc:
        mapping[key] = _SavedClusterConflict(str(exc))


def _load_run(root, read_clusters=True):
    """Union membership artifacts, but take cluster assignments from one source.

    The neighborhood table is authoritative when present. The original MMseqs2
    TSV supplies assignments only when that table is absent; its encoded member
    IDs may still recover pre-hash neighborhood membership in either case.
    """
    neighborhoods = defaultdict(dict)
    contextual, raw = {}, {}
    clustered = False

    def add(occ):
        neighborhoods[occ.neighborhood][occ.key] = occ

    table = root / "clust_res_in_neighborhoods" / "mmseqs_clust.tsv"
    use_neighborhood_table = table.is_file()
    if use_neighborhood_table:
        for number, row in enumerate(_rows(table, ("neighborhood_name", "locus_tag", "start", "strand")), 2):
            occ = _occurrence(row["neighborhood_name"], row["locus_tag"], row["start"], row["strand"])
            add(occ)
            rep = row.get("rep", "")
            if read_clusters and rep and rep.lower() != "nan":
                _record_saved_assignment(contextual, (occ.neighborhood, occ.key), rep, source=f"{table}:{number}")
                # Unclustered runs also write a table, with synthetic self-reps.
                clustered |= _protein_id(rep) != occ.protein

    membership = root / "neighborhoods_db.tsv"
    if membership.is_file():
        for row in _rows(membership, ("neighborhood_name", "locus_tag", "start", "strand")):
            add(_occurrence(row["neighborhood_name"], row["locus_tag"], row["start"], row["strand"]))

    fastas = sorted(p for p in root.glob("combined_fasta_partition*") if p.is_file())
    if not fastas and (root / "combined.fasta").is_file():
        fastas = [root / "combined.fasta"]
    for path in fastas:
        # Membership is entirely in headers; do not load protein sequences here.
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(">"):
                    identifier = line[1:].split()[0]
                    if "!!!" in identifier:
                        add(_encoded_occurrence(identifier))

    assignments = root / "combined_fastas_clust_res.tsv"
    read_original_clusters = read_clusters and not use_neighborhood_table
    if assignments.is_file():
        with assignments.open(encoding="utf-8", newline="") as handle:
            for number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if len(row) != 2 or not all(row):
                    raise ValueError(f"{assignments}:{number}: expected representative<TAB>member")
                rep, member = row
                if "!!!" in member:
                    occ = _encoded_occurrence(member)
                    add(occ)
                    if read_original_clusters:
                        _record_saved_assignment(contextual, (occ.neighborhood, occ.key), rep, source=f"{assignments}:{number}")
                elif read_original_clusters:
                    _record_saved_assignment(raw, _protein_id(member), rep, source=f"{assignments}:{number}")
                if read_original_clusters:
                    clustered = True

    if read_clusters:
        source = table if use_neighborhood_table else assignments
        LOG.info("Saved cluster source: %s", source if source.is_file() else "none")

    if not neighborhoods:
        raise ValueError(f"No extracted neighborhoods found in {root}; supply an extract_neighbors output directory")
    return neighborhoods, contextual, raw, clustered


def _select(available, names, n_random, seed):
    available = set(available)
    if isinstance(names, str):
        names = [names]
    names = list(names) if names is not None else []
    if any(not isinstance(name, str) for name in names):
        raise ValueError("neighborhood_names must be a string or a sequence of strings")
    selected = list(dict.fromkeys(names))
    missing = set(selected) - available
    if missing:
        LOG.warning("Skipping neighborhood names not found in the input directory: %s", ', '.join(sorted(missing)))
        selected = [name for name in selected if name in available]
        if not selected:
            raise ValueError("None of the requested neighborhood names were found in the input directory")
    n_random = (0 if selected else 10) if n_random is None else n_random
    if isinstance(n_random, bool) or not isinstance(n_random, int) or n_random < 0:
        raise ValueError("n_random must be a nonnegative integer")
    pool = sorted(available - set(selected))
    count = min(n_random, len(pool))
    if count < n_random:
        LOG.warning("Requested %d random neighborhoods; only %d remain. Including all %d.", n_random, count, count)
    selected.extend(random.Random(seed).sample(pool, count))
    if not selected:
        raise ValueError("No neighborhoods selected; provide names or a positive n_random")
    LOG.info("Selected %d explicit and %d random neighborhoods (%d total)", len(selected) - count, count, len(selected))
    return selected


def _sources(genomes, genome_tsv, required):
    if (genomes is None) == (genome_tsv is None):
        raise ValueError("Provide exactly one of genomes or genome_tsv for original GFF annotations")
    sources = {}
    if genome_tsv is not None:
        table = Path(genome_tsv).expanduser().resolve()
        with table.open(encoding="utf-8", newline="") as handle:
            for number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if len(row) != 3 or not all(row):
                    raise ValueError(f"{table}:{number}: expected genome<TAB>protein FASTA<TAB>GFF (no header)")
                name, protein, gff = row
                paths = [Path(p).expanduser() for p in (protein, gff)]
                pair = tuple(p if p.is_absolute() else table.parent / p for p in paths)
                if name in sources and sources[name] != pair:
                    raise ValueError(f"Ambiguous source mapping for genome {name}")
                sources[name] = pair
    else:
        directory = Path(genomes).expanduser()
        for name in required:
            # genome-only extraction may retain the _genomic suffix in names.
            gffs = [directory / f"{name}{suffix}" for suffix in (".gff", "_genomic.gff")]
            gffs = [p for p in gffs if p.is_file()]
            if len(gffs) != 1:
                raise ValueError(f"Expected one GFF for {name} in {directory}; found {len(gffs)}. Use genome_tsv to specify paths.")
            base = name.removesuffix("_genomic")
            fastas = list(dict.fromkeys(directory / f"{stem}{suffix}" for stem in (name, base)
                                       for suffix in ("_protein.faa", ".faa", ".fasta")))
            fastas = [p for p in fastas if p.is_file()]
            # Protein FASTAs are required only for reclustering.
            sources[name] = (fastas, gffs[0])
    missing = set(required) - sources.keys()
    if missing:
        raise ValueError(f"Missing source mappings for genomes: {', '.join(sorted(missing))}")
    for name in required:
        if not sources[name][1].is_file():
            raise ValueError(f"GFF not found for {name}: {sources[name][1]}")
    return {name: sources[name] for name in required}


def _gff_index(path, wanted):
    index = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"{path}:{number}: expected nine GFF columns")
            contig, _, kind, start, end, _, strand, _, attrs = fields
            if kind != "CDS":
                continue
            attributes = {}
            for attribute in attrs.split(";"):
                key, sep, value = attribute.partition("=")
                if sep:
                    attributes[unquote(key)] = unquote(value)
            identifiers = {_protein_id(attributes[key]) for key in ("protein_id", "locus_tag")
                           if attributes.get(key)}
            if not identifiers & wanted:
                continue
            try:
                start, end = int(start), int(end)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid CDS coordinates") from exc
            if start < 1 or end < start:
                raise ValueError(f"{path}:{number}: invalid CDS interval {start}-{end}")
            for identifier in identifiers & wanted:
                index[(contig, identifier, start, strand)].append({
                    "start": start, "end": end,
                    "product": attributes.get("product") or "No product annotation",
                })
    return index


def _make_tracks(neighborhoods, selected, sources):
    wanted = defaultdict(set)
    for name in selected:
        for occ in neighborhoods[name].values():
            wanted[occ.key[0]].add(occ.protein)
    indices = {genome: _gff_index(sources[genome][1], proteins) for genome, proteins in wanted.items()}
    tracks = []
    for name in selected:
        center, genome, contig, _ = _name_parts(name)
        genes = []
        for occ in sorted(neighborhoods[name].values(), key=lambda o: (o.start, o.protein, o.strand)):
            matches = indices[genome].get(occ.key[1:], [])
            if len(matches) != 1:
                raise ValueError(f"{'Missing' if not matches else 'Ambiguous'} GFF CDS match for {occ.protein} "
                                 f"in {name} at {occ.start} ({occ.strand}); found {len(matches)}")
            genes.append({**matches[0], "protein": occ.protein, "genome": genome, "contig": contig,
                          "strand": occ.strand, "anchor": occ.protein == _protein_id(center),
                          "cluster": None, "key": occ.key})
        anchors = [gene for gene in genes if gene["anchor"]]
        if len(anchors) != 1:
            raise ValueError(f"Expected one anchor {center} in {name}; found {len(anchors)}")
        anchor = anchors[0]
        midpoint = (anchor["start"] - 1 + anchor["end"]) / 2
        flip = anchor["strand"] == "-"
        for gene in genes:
            left, right = gene["start"] - 1 - midpoint, gene["end"] - midpoint
            gene["left"], gene["right"] = (-right, -left) if flip else (left, right)
            gene["direction"] = (1 if gene["strand"] == "+" else -1) * (-1 if flip else 1)
        lane_ends = []
        for gene in sorted(genes, key=lambda g: (g["left"], g["right"])):
            lane = next((i for i, end in enumerate(lane_ends) if end <= gene["left"]), len(lane_ends))
            if lane == len(lane_ends):
                lane_ends.append(gene["right"])
            else:
                lane_ends[lane] = gene["right"]
            gene["lane"] = lane
        tracks.append({"name": name, "genome": genome, "contig": contig, "flipped": flip,
                       "lanes": len(lane_ends), "genes": genes})
    return tracks


def _assign_saved(tracks, neighborhoods, contextual, raw, clustered):
    for track in tracks:
        for gene in track["genes"]:
            assignment = contextual.get((track["name"], gene["key"]), raw.get(gene["protein"]))
            if isinstance(assignment, _SavedClusterConflict):
                raise ValueError(f"{assignment.message}. This affects selected neighborhood {track['name']!r}; "
                                 "use --recluster to recompute assignments for the selected neighborhoods.")
    unused_conflicts = sum(isinstance(value, _SavedClusterConflict)
                           for mapping in (contextual, raw) for value in mapping.values())
    if unused_conflicts:
        LOG.warning("Skipped %d conflicting saved-assignment keys not used by the selected neighborhoods.", unused_conflicts)
    if not clustered:
        return "No verified MMseqs2 clustering found. Genes are unclustered; use --recluster to compare protein families."
    # A bare protein ID cannot disambiguate multiple genomes in old databases.
    locations = defaultdict(set)
    if raw:
        for members in neighborhoods.values():
            for occ in members.values():
                locations[occ.protein].add(occ.key)
    missing = 0
    occurrence_clusters = {}
    for track in tracks:
        for gene in track["genes"]:
            contextual_key = (track["name"], gene["key"])
            if contextual_key in contextual:
                gene["cluster"] = contextual[contextual_key]
            elif gene["protein"] in raw:
                if len(locations[gene["protein"]]) > 1:
                    raise ValueError(f"Ambiguous bare protein ID {gene['protein']} in saved clusters; use --recluster")
                gene["cluster"] = raw[gene["protein"]]
            else:
                missing += 1
            if gene["cluster"] is not None:
                _put_assignment(occurrence_clusters, gene["key"], gene["cluster"])
    return (f"Using saved MMseqs2 clusters. {missing} gene occurrences have no saved assignment and are unclustered."
            if missing else "Using saved MMseqs2 clusters.")


def _validate_settings(seq_id, cov, cov_mode, sensitivity, cluster_mode, threads, mem):
    for name, value in (("seq_id", seq_id), ("cov", cov)):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a fraction between 0 and 1")
    if (not isinstance(cov_mode, int) or cov_mode not in range(6)
            or (cluster_mode is not None and (not isinstance(cluster_mode, int) or cluster_mode not in range(4)))):
        raise ValueError("cov_mode must be 0–5 and cluster_mode must be 0–3")
    if sensitivity is not None and (not math.isfinite(sensitivity) or sensitivity <= 0):
        raise ValueError("sensitivity must be positive and finite")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not isinstance(mem, (float, int)) or not math.isfinite(mem) or mem <= 0:
        raise ValueError("mem must be positive and finite (GB)")


def _recluster(tracks, sources, directory, settings):
    if directory.exists():
        raise FileExistsError(f"Clustering directory already exists: {directory}")
    executable = shutil.which("mmseqs")
    if executable is None:
        raise ValueError("MMseqs2 executable 'mmseqs' was not found on PATH; install it to use --recluster")
    genes = {gene["key"]: gene for track in tracks for gene in track["genes"]}
    by_genome = defaultdict(set)
    for genome, _, protein, _, _ in genes:
        by_genome[genome].add(protein)
    sequences = {}
    for genome, wanted in by_genome.items():
        fasta = sources[genome][0]
        if isinstance(fasta, list):
            if len(fasta) != 1:
                raise ValueError(f"Expected one protein FASTA for {genome}; found {len(fasta)}. Use genome_tsv.")
            fasta = fasta[0]
        if not fasta.is_file():
            raise ValueError(f"Protein FASTA not found for {genome}: {fasta}")
        for identifier, sequence in _fasta(fasta):
            protein = _protein_id(identifier)
            if protein in wanted:
                key = (genome, protein)
                if key in sequences:
                    raise ValueError(f"Ambiguous protein FASTA identifier {protein} in {fasta}")
                if not sequence:
                    raise ValueError(f"Empty protein sequence {protein} in {fasta}")
                sequences[key] = sequence
        missing = wanted - {protein for source, protein in sequences if source == genome}
        if missing:
            raise ValueError(f"Missing protein sequences in {fasta}: {', '.join(sorted(missing))}")
    ids = {key: f"protein_{i:06d}" for i, key in enumerate(sorted(genes), 1)}
    directory.mkdir(parents=True, exist_ok=False)
    fasta = directory / "selected_proteins.fasta"
    with fasta.open("x", encoding="utf-8") as handle:
        for key, identifier in ids.items():
            handle.write(f">{identifier}\n{sequences[(key[0], key[2])]}\n")
    with (directory / "protein_mapping.tsv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("internal_id", "neighborhood_name", "genome", "contig", "protein_id", "start", "end", "strand"))
        for track in tracks:
            for gene in track["genes"]:
                writer.writerow((ids[gene["key"]], track["name"], gene["genome"], gene["contig"],
                                 gene["protein"], gene["start"], gene["end"], gene["strand"]))
    db, clusters, tmp = (str(directory / name) for name in ("proteins_db", "clusters_db", "tmp"))
    assignments = directory / "clusters.tsv"
    reps = str(directory / "representatives_db")
    flags = ["--min-seq-id", str(settings["seq_id"]), "-c", str(settings["cov"]),
             "--cov-mode", str(settings["cov_mode"]), "--similarity-type", "2",
             "--threads", str(settings["threads"]), "--split-memory-limit", f"{settings['mem']:g}G"]
    for key, flag in (("sensitivity", "-s"), ("cluster_mode", "--cluster-mode")):
        if settings[key] is not None:
            flags.extend((flag, str(settings[key])))
    commands = [[executable, "createdb", str(fasta), db],
                [executable, "cluster", db, clusters, tmp, *flags],
                [executable, "createtsv", db, db, clusters, str(assignments)],
                [executable, "createsubdb", clusters, db, reps],
                [executable, "convert2fasta", reps, str(directory / "representatives.fasta")]]
    metadata = {"settings": settings, "commands": commands, "status": "running",
                "neighborhoods": [track["name"] for track in tracks]}
    manifest = directory / "run.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        with (directory / "mmseqs.log").open("x", encoding="utf-8") as log:
            version = subprocess.run([executable, "version"], check=True, capture_output=True, text=True)
            metadata["version"] = version.stdout.strip()
            for command in commands:
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        mapping = {}
        valid_ids = set(ids.values())
        with assignments.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != 2 or row[0] not in valid_ids or row[1] not in valid_ids:
                    raise ValueError("MMseqs2 returned invalid representative/member IDs")
                _put_assignment(mapping, row[1], row[0])
        if set(mapping) != set(ids.values()):
            raise ValueError("MMseqs2 returned incomplete assignments for selected proteins")
        for track in tracks:
            for gene in track["genes"]:
                gene["cluster"] = mapping[ids[gene["key"]]]
        metadata["status"] = "complete"
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        raise ValueError(f"MMseqs2 reclustering failed: {exc}. See {directory / 'mmseqs.log'}") from exc
    finally:
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def viz_neighborhoods(input_dir, out, neighborhood_names=None, *, n_random=None, seed=None,
                      genomes=None, genome_tsv=None, recluster=False, cluster_dir=None,
                      seq_id=0.90, cov=0.90, cov_mode=0, sensitivity=None,
                      cluster_mode=None, threads=4, mem=8) -> Path:
    """Write ``out/neighborhoods.html`` and return its path.

    ``neighborhood_names`` is an exact name or sequence of names from extraction.
    Missing names are skipped with a warning. If no requested names are found,
    raise ``ValueError`` even when ``n_random`` is positive.
    ``n_random`` adds random names, excluding explicit selections; its default is
    ten without names and zero with names. ``seed`` makes sampling reproducible.
    Supply either ``genomes`` (a directory) or ``genome_tsv`` (headerless columns
    genome, protein FASTA, GFF; relative paths resolve against that TSV).
    Saved cluster assignments come from ``clust_res_in_neighborhoods/mmseqs_clust.tsv``
    when present, otherwise ``combined_fastas_clust_res.tsv``. Assignments from
    these sources are never mixed; missing assignments remain unclustered.
    Reclustering uses original sequences from selected neighborhoods only, and
    writes a new ``cluster_dir`` (default: ``out/recluster``). Identity and coverage
    are fractions. ``mem`` is the MMseqs2 split-memory limit in GB. No existing
    output HTML or clustering directory is overwritten.
    """
    root, output = Path(input_dir).expanduser().resolve(), Path(out).expanduser().resolve()
    html = output / "neighborhoods.html"
    if not root.is_dir():
        raise ValueError(f"Extraction directory does not exist: {root}")
    if html.exists():
        raise FileExistsError(f"Visualization already exists: {html}")
    if cluster_dir is not None and not recluster:
        raise ValueError("cluster_dir requires recluster=True")
    _validate_settings(seq_id, cov, cov_mode, sensitivity, cluster_mode, threads, mem)
    neighborhoods, contextual, raw, clustered = _load_run(root, read_clusters=not recluster)
    selected = _select(neighborhoods, neighborhood_names, n_random, seed)
    sources = _sources(genomes, genome_tsv, {_name_parts(name)[1] for name in selected})
    tracks = _make_tracks(neighborhoods, selected, sources)
    settings = dict(seq_id=seq_id, cov=cov, cov_mode=cov_mode, sensitivity=sensitivity,
                    cluster_mode=cluster_mode, threads=threads, mem=mem)
    if recluster:
        directory = Path(cluster_dir).expanduser().resolve() if cluster_dir is not None else output / "recluster"
        _recluster(tracks, sources, directory, settings)
        message = "Reclustered proteins from selected neighborhoods with MMseqs2."
    else:
        message = _assign_saved(tracks, neighborhoods, contextual, raw, clustered)
    LOG.info(message)
    cluster_tracks = defaultdict(set)
    for track in tracks:
        for gene in track["genes"]:
            if gene["cluster"] is not None:
                cluster_tracks[gene["cluster"]].add(track["name"])
            del gene["key"]
    shared = sorted(cluster for cluster, names in cluster_tracks.items() if len(names) >= 2)
    data = {"tracks": tracks, "shared_clusters": shared, "message": message,
            "seed": seed, "recluster_settings": settings if recluster else None}
    # Escape '<' to prevent GFF text from terminating the JSON script element.
    payload = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = HTML.replace("__NEIGHBORHOOD_DATA__", payload)
    output.mkdir(parents=True, exist_ok=True)
    with html.open("x", encoding="utf-8") as handle:
        handle.write(document)
    return html
