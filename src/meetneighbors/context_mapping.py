"""Disk-backed label and occurrence mapping for context-only inference.

No embedding matrices or hash-group Cartesian products are materialized here.
The saved hash is used only to recover retained neighborhoods, not as a model
input identity (the historical hash does not include strands).
"""

import csv
import json
import math
from pathlib import Path
import pickle
import sqlite3


def find_neighborhood_run(input_dir, explicit=None):
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Neighborhood run is not a directory: {root}")
        return root
    root = Path(input_dir).resolve()
    for candidate in (root, *root.parents):
        if any((candidate / name).is_file() for name in
               ("glm_embeds.tsv", "nnhash_mapping.obj", "clust_res_in_neighborhoods/mmseqs_clust.tsv")):
            return candidate
    # Ordinary, non-hashed runs can reconstruct the legacy labels from inputs.
    return root


def _queries(value):
    """Match the legacy query explode, excluding absent labels."""
    for query in value if isinstance(value, (list, tuple)) else [value]:
        if query is None or (isinstance(query, float) and math.isnan(query)):
            continue
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Invalid query label in neighborhood mapping: {query!r}")
        yield query


class ContextIndex:
    def __init__(self, path):
        self.hashed = False
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-16384;
            CREATE TABLE inputs (
                id INTEGER PRIMARY KEY, file TEXT, row INTEGER, name TEXT,
                proteins TEXT, center INTEGER, hash TEXT);
            CREATE INDEX inputs_name ON inputs(name);
            CREATE INDEX inputs_hash ON inputs(hash);
            CREATE TABLE positions (input_id INTEGER, rep TEXT, position INTEGER, flat_index INTEGER,
                PRIMARY KEY(input_id, position));
            CREATE INDEX positions_rep ON positions(input_id, rep);
            CREATE TABLE names (name TEXT PRIMARY KEY, hash TEXT);
            CREATE TABLE labels (id INTEGER PRIMARY KEY, hash TEXT, name TEXT, rep TEXT, query TEXT);
            CREATE INDEX labels_name ON labels(name,id);
            CREATE INDEX labels_hash_rep ON labels(hash,rep);
            CREATE TABLE reconstructed (id INTEGER PRIMARY KEY, name TEXT, rep TEXT, query TEXT,
                UNIQUE(name,rep,query));
            CREATE TABLE occurrences (hash TEXT, rep TEXT, input_id INTEGER, position INTEGER,
                PRIMARY KEY(hash,rep));
            CREATE TABLE targets (id INTEGER PRIMARY KEY, name TEXT, rep TEXT, query TEXT,
                job_id INTEGER, UNIQUE(name, rep, query));
            CREATE TABLE jobs (id INTEGER PRIMARY KEY, input_id INTEGER, center INTEGER,
                signature BLOB, UNIQUE(input_id, center));
            CREATE TABLE vectors (signature BLOB PRIMARY KEY, job_id INTEGER, value BLOB);
        """)

    def close(self):
        self.db.close()

    def add_inputs(self, rows, input_mode):
        source_file, offset = None, 0
        for row in rows:
            if source_file != row.source_file:
                source_file, offset = row.source_file, 0
            proteins = list(row.proteins)
            if input_mode == "masked":
                proteins[row.center_index] = proteins[row.center_index][0] + row.center_id
            cursor = self.db.execute(
                "INSERT INTO inputs(file,row,name,proteins,center) VALUES (?,?,?,?,?)",
                (row.source_file, row.source_row, row.name, json.dumps(proteins), row.center_index))
            self.db.executemany("INSERT INTO positions VALUES (?,?,?,?)",
                                ((cursor.lastrowid, p[1:], i, offset + i) for i, p in enumerate(proteins)))
            offset += len(proteins)
        self.db.commit()
        if not self.db.execute("SELECT 1 FROM inputs LIMIT 1").fetchone():
            raise ValueError("Input TSVs contain no neighborhoods")

    def load_hashes(self, path):
        if not path.is_file():
            return False
        with path.open("rb") as handle:
            mapping = pickle.load(handle)
        if not isinstance(mapping, dict):
            raise ValueError(f"{path}: expected a neighborhood hash dictionary")
        hashed = bool(mapping)
        self.hashed = hashed
        for hash_value, entries in mapping.items():
            hash_value = str(hash_value)
            for entry in entries:
                if len(entry) != 3:
                    raise ValueError(f"{path}: expected (query, rep, neighborhood_name)")
                query, rep, name = entry
                if not isinstance(name, str) or not name or not isinstance(rep, str) or not rep:
                    raise ValueError(f"{path}: invalid neighborhood or representative")
                previous = self.db.execute("SELECT hash FROM names WHERE name=?", (name,)).fetchone()
                if previous is not None and previous[0] != hash_value:
                    raise ValueError(f"{path}: conflicting hashes for {name}")
                self.db.execute("INSERT OR IGNORE INTO names VALUES (?,?)", (name, hash_value))
                self.db.executemany("INSERT INTO labels(hash,name,rep,query) VALUES (?,?,?,?)",
                                    ((hash_value, name, rep, q) for q in _queries(query)))
        del mapping
        self.db.execute("UPDATE inputs SET hash=(SELECT hash FROM names WHERE names.name=inputs.name)")
        self.db.commit()
        return hashed

    def load_manifest(self, run, hashed):
        if hashed:
            self._reconstruct_labels()
        reference = run / "glm_embeds.tsv"
        if reference.is_file():
            with reference.open(newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                columns = next(reader, [])
                if not {"neighborhood_name", "query"}.issubset(columns):
                    raise ValueError(f"{reference}: missing neighborhood_name/query columns")
                # Avoid building 1,280 embedding-column dictionary entries per row.
                metadata = {key: columns.index(key) for key in
                            ("neighborhood_name", "query", "rep", "rep_y") if key in columns}
                for line, row in enumerate(reader, 2):
                    if len(row) <= max(metadata.values()):
                        raise ValueError(f"{reference}:{line}: truncated metadata row")
                    values = {key: row[i] for key, i in metadata.items()}
                    name, query = values["neighborhood_name"], values["query"]
                    rep = values.get("rep", values.get("rep_y"))
                    if "rep" in values and "rep_y" in values and values["rep"] != values["rep_y"]:
                        raise ValueError(f"{reference}:{line}: conflicting rep and rep_y")
                    if rep is None:
                        # Ordinary legacy tables omit rep; use the explicit center.
                        reps = self.db.execute("""SELECT DISTINCT p.rep FROM inputs i JOIN positions p
                            ON p.input_id=i.id AND p.position=i.center WHERE i.name=?""", (name,)).fetchall()
                        if len(reps) != 1:
                            raise ValueError(f"{reference}:{line}: cannot recover an unambiguous center rep")
                        rep = reps[0][0]
                    if not all(isinstance(x, str) and x.strip() for x in (name, rep, query)):
                        raise ValueError(f"{reference}:{line}: missing neighborhood, rep or query")
                    self.db.execute("INSERT OR IGNORE INTO targets(name,rep,query) VALUES (?,?,?)",
                                    (name, rep, query))
        elif hashed:
            self.db.execute("""INSERT INTO targets(name,rep,query)
                SELECT name,rep,query FROM reconstructed ORDER BY id""")
        else:
            # This is get_glm_embeddf's ordinary-run filename convention.
            for row in self.db.execute("""SELECT i.name,i.file,p.rep FROM inputs i JOIN positions p
                    ON p.input_id=i.id AND p.position=i.center ORDER BY i.id"""):
                query = Path(row["file"]).stem.split("!!!")[0]
                self.db.execute("INSERT OR IGNORE INTO targets(name,rep,query) VALUES (?,?,?)",
                                (row["name"], row["rep"], query))
        self.db.commit()
        if not self.db.execute("SELECT 1 FROM targets LIMIT 1").fetchone():
            raise ValueError("No labeled GLM metadata combinations found")
        if hashed and reference.is_file():
            # Compare independently reconstructed labels to the reference, rather
            # than trusting arbitrary query labels merely because they are in it.
            missing = self.db.execute("""SELECT name,rep,query FROM targets EXCEPT
                SELECT name,rep,query FROM reconstructed LIMIT 3""").fetchall()
            extra = self.db.execute("""SELECT name,rep,query FROM reconstructed EXCEPT
                SELECT name,rep,query FROM targets LIMIT 3""").fetchall()
            if missing or extra:
                raise ValueError(f"GLM metadata mismatch against {reference}: "
                                 f"unreconstructed reference rows {[tuple(r) for r in missing]}; "
                                 f"unexpected reconstructed rows {[tuple(r) for r in extra]}. "
                                 "Check that inputs, hash mapping and reference belong to the same complete run.")

    def _reconstruct_labels(self):
        # Equivalent to get_hashedout_embeds: recover original names by hash,
        # match rep occurrences, then attach queries by (name, rep). Avoid its
        # hash-group Cartesian product with indexed EXISTS lookups instead.
        # Its first deduplication retains the first recorded rep for each name;
        # its final deduplication retains one row per (query, neighborhood).
        self.db.execute("""INSERT INTO reconstructed(name,rep,query)
            SELECT name,rep,query FROM (
                SELECT l.*, ROW_NUMBER() OVER (PARTITION BY name,query ORDER BY l.id) AS rank
                FROM labels l WHERE l.rep=(
                    SELECT first.rep FROM labels first WHERE first.name=l.name ORDER BY first.id LIMIT 1)
                AND EXISTS (
                    SELECT 1 FROM inputs i JOIN positions p ON p.input_id=i.id
                    WHERE i.hash=l.hash AND p.rep=l.rep))
            WHERE rank=1 ORDER BY id""")

    def _hashed_occurrence(self, target):
        hash_row = self.db.execute("SELECT hash FROM names WHERE name=?", (target["name"],)).fetchone()
        if hash_row is None:
            return None
        key = (hash_row[0], target["rep"])
        cached = self.db.execute("SELECT input_id,position FROM occurrences WHERE hash=? AND rep=?", key).fetchone()
        if cached is not None:
            return tuple(cached)
        # The source center is deliberately absent: it belongs to the retained
        # neighborhood and cannot identify an occurrence in a restored label.
        candidates = self.db.execute("""SELECT i.id,i.proteins,MIN(p.position) AS position
            FROM inputs i JOIN positions p ON p.input_id=i.id
            WHERE i.hash=? AND p.rep=? GROUP BY i.id ORDER BY i.id""", key).fetchall()
        if not candidates:
            return None
        first = candidates[0]
        if any((r["proteins"], r["position"]) != (first["proteins"], first["position"])
               for r in candidates[1:]):
            raise ValueError(f"Conflicting retained contexts for {target['name']} / {target['rep']}")
        # get_hashedout_embeds keeps the first matching flattened occurrence
        # when it drops duplicate query/neighborhood rows (including duplicates
        # with opposite strands). Preserve that choice, not the old VF center.
        result = (first["id"], first["position"])
        self.db.execute("INSERT INTO occurrences VALUES (?,?,?,?)", (*key, *result))
        return result

    def resolve(self):
        for target in self.db.execute("SELECT * FROM targets ORDER BY id"):
            if self.hashed:
                occurrence = self._hashed_occurrence(target)
                if occurrence is None:
                    raise ValueError(f"Cannot resolve GLM label {target['name']} / {target['rep']} / {target['query']}")
                self._assign_job(target["id"], *occurrence)
                continue
            candidates = self.db.execute("SELECT * FROM inputs WHERE name=? ORDER BY id",
                                         (target["name"],)).fetchall()
            direct = bool(candidates)
            if not direct:
                candidates = self.db.execute("""SELECT i.* FROM inputs i JOIN names n
                    ON i.hash=n.hash WHERE n.name=? ORDER BY i.id""", (target["name"],)).fetchall()
            resolved = []
            for candidate in candidates:
                positions = [r[0] for r in self.db.execute(
                    "SELECT position FROM positions WHERE input_id=? AND rep=? ORDER BY position",
                    (candidate["id"], target["rep"]))]
                # Ordinary, non-hashed inputs retain their own explicit center.
                if candidate["center"] in positions:
                    position = candidate["center"]
                elif len(positions) == 1:
                    position = positions[0]
                elif not positions:
                    continue
                else:
                    raise ValueError(f"{candidate['file']}:{candidate['row']}: ambiguous occurrences of "
                                     f"{target['rep']} at zero-based positions {positions}; "
                                     f"VF center index {candidate['center']} is not among them, "
                                     f"for {target['name']} / {target['query']}")
                resolved.append((candidate, position))
            if not resolved:
                raise ValueError(f"Cannot resolve GLM label {target['name']} / {target['rep']} / "
                                 f"{target['query']}; check --neighborhood-run and input coverage")
            candidate, position = resolved[0]
            if any((r["proteins"], p) != (candidate["proteins"], position) for r, p in resolved[1:]):
                raise ValueError(f"Conflicting retained contexts for {target['name']} / {target['rep']}")
            self._assign_job(target["id"], candidate["id"], position)
        self.db.commit()

    def _assign_job(self, target_id, input_id, position):
        self.db.execute("INSERT OR IGNORE INTO jobs(input_id,center) VALUES (?,?)", (input_id, position))
        job = self.db.execute("SELECT id FROM jobs WHERE input_id=? AND center=?",
                              (input_id, position)).fetchone()[0]
        self.db.execute("UPDATE targets SET job_id=? WHERE id=?", (job, target_id))

    def iter_jobs(self, unique=False):
        from .context_only import Neighborhood

        join = "JOIN vectors v ON v.job_id=j.id" if unique else ""
        for row in self.db.execute(f"""SELECT j.id AS job_id,j.signature,i.*,j.center AS target_center
                FROM jobs j JOIN inputs i ON i.id=j.input_id {join} ORDER BY j.id"""):
            proteins = tuple(json.loads(row["proteins"]))
            center = row["target_center"]
            yield row["job_id"], row["signature"], Neighborhood(
                row["file"], row["row"], row["name"], proteins, center, proteins[center][1:])

    def output_rows(self):
        return self.db.execute("""SELECT i.file,i.row,t.name,t.rep,t.query,v.value
            FROM targets t JOIN jobs j ON j.id=t.job_id JOIN inputs i ON i.id=j.input_id
            JOIN vectors v ON v.signature=j.signature ORDER BY t.id""")
