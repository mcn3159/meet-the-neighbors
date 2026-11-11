"""Small CPU fixtures for label parity, occurrence mapping, and context reuse."""

import csv
import json
import logging
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from meetneighbors import context_only as co
from meetneighbors.context_mapping import ContextIndex, find_neighborhood_run


class ContextMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "glm_inputs").mkdir()
        (self.root / "glm_outputs").mkdir()
        self.path = self.root / "glm_inputs" / "qfile.tsv"
        self.path.write_text("retained\t+A;-B;+C\t0\n")
        self.embeddings = {p: np.full(1280, n, dtype=np.float16)
                           for p, n in zip("ABCD", [2, 4, 6, 8])}
        self.mean, self.std = np.full(1280, 3.0), np.full(1280, 2.0)
        with (self.root / "glm_outputs" / co.ESM_FILENAME).open("wb") as f:
            pickle.dump(list(self.embeddings.items()), f)

    def manifest(self, rows, rep_column="rep_y"):
        with (self.root / "glm_embeds.tsv").open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["neighborhood_name", rep_column, "query", "0"])
            writer.writerows([*r, 12345] for r in rows)

    def hashes(self, mapping):
        with (self.root / "nnhash_mapping.obj").open("wb") as f:
            pickle.dump(mapping, f)

    def index(self, mode="original"):
        index = ContextIndex(self.root / "index.sqlite")
        self.addCleanup(index.close)
        index.add_inputs(co.iter_neighborhoods([self.path], mode), mode)
        hashed = index.load_hashes(self.root / "nnhash_mapping.obj")
        index.load_manifest(self.root, hashed)
        index.resolve()
        return index

    def run_fixture(self, mode="original", input_dir=None):
        calls = []

        def embed(model, inputs, attention, rows, device):
            calls.extend((r, inputs[i].clone()) for i, r in enumerate(rows))
            return np.stack([np.full(1280, float(inputs[i, :, :1280].sum()) + r.center_index)
                             for i, r in enumerate(rows)])

        with patch.object(co, "load_normalization", return_value=(self.mean, self.std)), \
                patch.object(co, "load_model") as load, patch.object(co, "embed_centers", side_effect=embed):
            result = co.run_context_only(input_dir or self.root, self.root / "out", batch_size=2,
                                         device="cpu", input_mode=mode, neighborhood_run=self.root)
            load.assert_called_once()
        with result.open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(list((self.root / "out").glob(".context_only_*")), [])
        return rows, calls

    def test_duplicate_centers_from_explicit_manifest(self):
        self.path.write_text("retained\t+A;-B;+A\t0\n")
        (self.root / "glm_embeds.tsv").write_text(
            "neighborhood_name\trep\tquery\tnn_center_id\t0\n"
            "retained\tA\tq\t0\t99\nretained\tA\tq\t2\t99\n")
        rows, calls = self.run_fixture()
        self.assertEqual([row['nn_center_id'] for row in rows], ['0', '2'])
        self.assertEqual([row['query'] for row in rows], ['q', 'q'])
        self.assertEqual([row.center_index for row, _ in calls], [0, 2])
        for row, inputs in calls:
            np.testing.assert_array_equal(inputs[row.center_index, :1280], -2)
            np.testing.assert_array_equal(inputs[2 - row.center_index, :1280], -.5)

    def test_duplicate_centers_from_indexed_hash(self):
        self.path.write_text("retained\t+A;-B;+A\t0\n")
        self.hashes({7: [('q', 'A', 'restored', 0), ('q', 'A', 'restored', 2),
                        ('retained-q', 'A', 'retained', 0)]})
        rows, calls = self.run_fixture()
        restored = [r for r in rows if r['neighborhood_name'] == 'restored']
        self.assertEqual([r['nn_center_id'] for r in restored], ['0', '2'])
        self.assertEqual([row.center_index for row, _ in calls], [0, 2])

    def test_indexed_duplicate_manifest_requires_every_position(self):
        self.path.write_text("retained\t+A;-B;+A\t0\n")
        self.hashes({7: [('q', 'A', 'retained', 0), ('q', 'A', 'retained', 2)]})
        (self.root / "glm_embeds.tsv").write_text(
            "neighborhood_name\trep\tquery\tnn_center_id\nretained\tA\tq\t0\n")
        with self.assertRaisesRegex(ValueError, 'unexpected reconstructed rows'):
            self.index()

    def test_indexed_duplicate_manifest_without_position_is_ambiguous(self):
        self.path.write_text("retained\t+A;-B;+A\t0\n")
        self.hashes({7: [('q', 'A', 'retained', 0), ('q', 'A', 'retained', 2)]})
        self.manifest([('retained', 'A', 'q')])
        with self.assertRaisesRegex(ValueError, 'nn_center_id is required'):
            self.index()

    def test_invalid_explicit_center_is_rejected_before_inference(self):
        (self.root / "glm_embeds.tsv").write_text(
            "neighborhood_name\trep\tquery\tnn_center_id\nretained\tA\tq\t1\n")
        with patch.object(co, 'load_model') as model, self.assertRaisesRegex(ValueError, 'Cannot resolve'):
            co.run_context_only(self.root, self.root / 'out', input_mode='original')
        model.assert_not_called()

    def test_full_manifest_and_different_target_restoration(self):
        expected = [("retained", "A", "same"), ("omitted1", "A", "same"),
                    ("omitted1", "A", "different"), ("omitted2", "B", "other")]
        self.hashes({7: [(q, r, n) for n, r, q in expected]})
        self.manifest([*expected, expected[0]])
        masked = self.root / "masked_embeds"
        (masked / "glm_inputs").mkdir(parents=True)
        (masked / "glm_inputs" / "qfile.tsv").write_text("retained\t+A_masked;-B;+C\t0\n")
        rows, calls = self.run_fixture("masked", masked)
        self.assertEqual([(r["neighborhood_name"], r["rep"], r["query"]) for r in rows], expected)
        self.assertEqual(len(calls), 2)
        self.assertEqual([r[0].center_index for r in calls], [0, 1])
        np.testing.assert_array_equal(calls[1][1][0, :1280], -0.5)  # Original A restored.
        np.testing.assert_array_equal(calls[1][1][1, :1280], -2)  # Only B masked.
        self.assertEqual(calls[1][1][1, -1], -0.5)
        self.assertTrue(all(r["source_file"] == "qfile.tsv" and r["source_row"] == "1" for r in rows))
        for r in rows[1:3]:
            np.testing.assert_allclose([float(r[str(i)]) for i in range(1280)],
                                       [float(rows[0][str(i)]) for i in range(1280)], rtol=1e-3, atol=5e-3)
        self.assertNotEqual(rows[0]["0"], rows[3]["0"])

    def test_different_center_identity_shares_inference_but_strands_do_not(self):
        # An ordinary run may retain all variants. A hashed run would retain
        # only one template per hash, so do not invent a conflicting hash group.
        self.path.write_text("one\t+A;-B;+C\t0\ntwo\t+D;-B;+C\t0\nthree\t-D;-B;+C\t0\n")
        self.manifest([("one", "A", "q1"), ("two", "D", "q2"), ("three", "D", "q2")], "rep")
        rows, calls = self.run_fixture()
        self.assertEqual(len(calls), 2)
        self.assertEqual(rows[0]["0"], rows[1]["0"])
        self.assertEqual(rows[1]["source_row"], "2")

    def legacy_fixture(self):
        result_dir = self.root / "glm_outputs" / "qfile" / "results" / "results"
        result_dir.mkdir(parents=True)
        with (result_dir / "batch.pkl.glm.embs.pkl").open("wb") as f:
            pickle.dump([(p, self.embeddings[p]) for p in "ABC"], f)

    def test_hash_fallback_matches_existing_assembly(self):
        from meetneighbors.compute_umap import unpack_embeddings, get_glm_embeddf
        import pandas as pd

        mapping = {7: [("q1", "A", "retained"), (["q1", "q2"], "A", "omitted"),
                       ("legacy_deduplicated", "B", "omitted"),
                       ("q3", "B", "other"), ("q3", "B", "other")]}
        self.hashes(mapping)
        self.legacy_fixture()
        clusters = pd.DataFrame([dict(neighborhood_name="retained", rep="A", query="q1",
                                     prot_gffname="A!!!g", VF_center="A", gff="g", nn_hashes=7)])
        legacy = get_glm_embeddf(unpack_embeddings(
            str(self.root / "glm_outputs"), str(self.root / "glm_inputs"), clusters,
            logging.getLogger(__name__), mem_optimize=mapping), mem_optimize=True)
        expected = set(legacy[["neighborhood_name", "rep_y", "query"]].itertuples(index=False, name=None))
        index = self.index()
        self.assertEqual({tuple(r) for r in index.db.execute("SELECT name,rep,query FROM targets")}, expected)

    def test_ordinary_fallback_matches_existing_assembly(self):
        from meetneighbors.compute_umap import unpack_embeddings, get_glm_embeddf
        import pandas as pd

        self.legacy_fixture()
        clusters = pd.DataFrame([dict(neighborhood_name="retained", rep="A", query="qfile")])
        legacy = get_glm_embeddf(unpack_embeddings(
            str(self.root / "glm_outputs"), str(self.root / "glm_inputs"), clusters, logging.getLogger(__name__)))
        index = self.index()
        self.assertEqual({tuple(r) for r in index.db.execute("SELECT name,query FROM targets")},
                         set(legacy[["neighborhood_name", "query"]].itertuples(index=False, name=None)))
        self.assertEqual(index.db.execute("SELECT rep FROM targets").fetchone()[0], "A")

    def test_direct_explicit_center_resolves_repeated_rep(self):
        self.path.write_text("retained\t+A;-B;+A\t2\n")
        self.manifest([("retained", "A", "q")])
        self.assertEqual(next(self.index().iter_jobs())[2].center_index, 2)

    def test_omitted_repeated_rep_matches_legacy_occurrence(self):
        self.path.write_text("retained\t+A;-B;+A\t2\n")
        mapping = {7: [("q", "A", "retained"), ("q", "A", "omitted")]}
        self.hashes(mapping)
        expected = self.legacy_occurrences(mapping)
        self.manifest(expected)
        rows, calls = self.run_fixture()
        self.assertEqual([(r["neighborhood_name"], r["rep"], r["query"]) for r in rows], list(expected))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].center_index, expected[("omitted", "A", "q")])
        np.testing.assert_array_equal(calls[0][1][0, :1280], -2)
        np.testing.assert_array_equal(calls[0][1][2, :1280], -0.5)

    def test_omitted_repeated_rep_uses_masked_vf_center(self):
        self.path.write_text("retained\t+A_masked;-B;+A\t0\n")
        self.hashes({7: [("q", "A", "retained"), ("q", "A", "omitted")]})
        self.manifest([("retained", "A", "q"), ("omitted", "A", "q")])
        _, calls = self.run_fixture(mode="masked")
        self.assertEqual(calls[0][0].center_index, 0)
        np.testing.assert_array_equal(calls[0][1][0, :1280], -2)
        np.testing.assert_array_equal(calls[0][1][2, :1280], -0.5)

    def test_omitted_repeated_noncenter_rep_matches_legacy_occurrence(self):
        self.path.write_text("retained\t+A;-B;+A\t1\n")
        mapping = {7: [("q", "B", "retained"), ("q", "A", "omitted")]}
        self.hashes(mapping)
        expected = self.legacy_occurrences(mapping)
        self.manifest(expected)
        index = self.index()
        self.assertEqual(self.resolved_occurrences(index), expected)

    def legacy_occurrences(self, mapping):
        """Use the actual legacy assembler as an independent position oracle."""
        from meetneighbors.compute_umap import unpack_embeddings, get_glm_embeddf
        import pandas as pd

        rows = list(co.iter_neighborhoods([self.path], "original"))
        root = self.root / "legacy"
        (root / "inputs").mkdir(parents=True)
        results = root / "outputs" / "qfile" / "results" / "results"
        results.mkdir(parents=True)
        (root / "inputs" / "qfile.tsv").write_text(self.path.read_text())
        vectors, centers = [], []
        name_hash = {name: h for h, entries in mapping.items() for _, _, name in entries}
        for n, row in enumerate(rows):
            for protein in row.proteins:
                vectors.append((protein[1:], np.full(1280, len(vectors), dtype=np.float16)))
            centers.append(dict(neighborhood_name=row.name, rep=row.center_id, query="fixture",
                                prot_gffname=f"center{n}!!!g", VF_center=f"center{n}", gff="g",
                                nn_hashes=name_hash[row.name]))
        with (results / "batch.pkl.glm.embs.pkl").open("wb") as f:
            pickle.dump(vectors, f)
        legacy = get_glm_embeddf(unpack_embeddings(
            str(root / "outputs"), str(root / "inputs"), pd.DataFrame(centers),
            logging.getLogger(__name__), mem_optimize=mapping), mem_optimize=True)
        return {(r["neighborhood_name"], r["rep_y"], r["query"]): int(r[0])
                for _, r in legacy.iterrows()}

    @staticmethod
    def resolved_occurrences(index):
        return {tuple(r)[:3]: r["flat_index"] for r in index.db.execute("""
            SELECT t.name,t.rep,t.query,p.flat_index FROM targets t
            JOIN jobs j ON j.id=t.job_id JOIN positions p
            ON p.input_id=j.input_id AND p.position=j.center""")}

    def test_reference_query_mismatch_fails_before_inference(self):
        self.hashes({7: [("correct", "A", "retained")]})
        self.manifest([("retained", "A", "incorrect")])
        with patch.object(co, "load_model") as model, self.assertRaisesRegex(ValueError, "GLM metadata mismatch"):
            co.run_context_only(self.root, self.root / "out", input_mode="original")
        model.assert_not_called()

    def test_missing_and_extra_reference_combinations_are_rejected(self):
        self.hashes({7: [("q1", "A", "retained"), ("q2", "A", "omitted")]})
        self.manifest([("retained", "A", "q1")])
        with self.assertRaisesRegex(ValueError, "unexpected reconstructed rows"):
            self.index()

    def test_reported_duplicate_cases_match_legacy_positions_and_labels(self):
        cases = json.loads((Path(__file__).parent / "test_data/context_only_duplicate_cases.json").read_text())
        mapping = {}
        rows = []
        offset = 0
        expected_targets = {}
        for i, case in enumerate(cases):
            proteins = case["proteins"]
            rows.append([case["source_neighborhood"], ";".join(proteins), case["vf_center_index"]])
            # Only these two source rows are real-data fixtures. The small hash
            # groups reproduce the reported retained/restored relationships;
            # no production pickle or embeddings are loaded.
            mapping[i] = [(f"source_fixture_{i}", proteins[case["vf_center_index"]][1:], case["source_neighborhood"]),
                          (case["query"], case["rep"], case["target_neighborhood"])]
            expected_targets[(case["target_neighborhood"], case["rep"], case["query"])] = offset + case["legacy_position"]
            offset += len(proteins)
            for j, protein in enumerate(proteins):
                self.embeddings.setdefault(protein[1:], np.full(1280, j + 10, dtype=np.float16))
        with self.path.open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)
        self.hashes(mapping)
        oracle = self.legacy_occurrences(mapping)
        for target, position in expected_targets.items():
            self.assertEqual(oracle[target], position)
        self.manifest(oracle)
        index = self.index()
        self.assertEqual(self.resolved_occurrences(index), oracle)
        jobs = {job_id: row for job_id, _, row in index.iter_jobs()}
        for case in cases:
            job_id = index.db.execute("SELECT job_id FROM targets WHERE name=? AND rep=? AND query=?",
                                      (case["target_neighborhood"], case["rep"], case["query"])).fetchone()[0]
            job = jobs[job_id]
            self.assertEqual(job.center_index, case["legacy_position"])
            inputs, _ = co.prepare_batch([job], self.embeddings, self.mean, self.std)
            for position, protein in enumerate(job.proteins):
                if protein[1:] != case["rep"]:
                    continue
                expected = (-np.ones(1280) if position == case["legacy_position"]
                            else self.embeddings[case["rep"]])
                np.testing.assert_array_equal(inputs[0, position, :1280], ((expected-self.mean)/self.std).astype(np.float16))

    def test_conflicting_retained_contexts_in_one_hash_are_rejected(self):
        self.path.write_text("one\t+A;-B;+C\t0\ntwo\t-A;-B;+C\t0\n")
        self.hashes({7: [("q", "A", "one"), ("q", "A", "two")]})
        self.manifest([("one", "A", "q"), ("two", "A", "q")])
        with self.assertRaisesRegex(ValueError, "Conflicting retained contexts"):
            self.index()

    def test_missing_mapping_fails_without_output(self):
        self.manifest([("absent", "A", "q")])
        with patch.object(co, "load_model") as model, self.assertRaisesRegex(ValueError, "Cannot resolve GLM label"):
            co.run_context_only(self.root, self.root / "out", input_mode="original")
        model.assert_not_called()
        self.assertFalse((self.root / "out" / co.OUTPUT_FILENAME).exists())

    def test_missing_original_center_vector_is_not_left_masked(self):
        self.path.write_text("retained\t+A_masked;-B;+C\t0\n")
        self.manifest([("retained", "B", "q")])
        entries = [(k, v) for k, v in self.embeddings.items() if k != "A"]
        entries.append(("A_masked", np.full(1280, -1)))
        with (self.root / "glm_outputs" / co.ESM_FILENAME).open("wb") as f:
            pickle.dump(entries, f)
        with patch.object(co, "load_model") as model, \
                patch.object(co, "load_normalization", return_value=(self.mean, self.std)), \
                self.assertRaisesRegex(ValueError, "missing neighbor ESM embedding A"):
            co.run_context_only(self.root, self.root / "out", input_mode="masked")
        model.assert_not_called()

    def test_conflicting_reference_columns_fail(self):
        (self.root / "glm_embeds.tsv").write_text("neighborhood_name\trep\trep_y\tquery\nretained\tA\tB\tq\n")
        with self.assertRaisesRegex(ValueError, "conflicting rep.*rep .*='A'.*rep_y .*='B'"):
            self.index()

    def test_empty_reference_query_reports_context_before_inference(self):
        self.hashes({7: [("known-query", "A", "retained")]})
        self.manifest([("retained", "A", "")])
        with patch.object(co, "load_model") as model, self.assertRaises(ValueError) as raised:
            co.run_context_only(self.root, self.root / "out", input_mode="original")
        message = str(raised.exception)
        for expected in ("glm_embeds.tsv:2:", "['query']", "4/4 fields",
                         "neighborhood_name (column 1)='retained'", "rep_y (column 2)='A'",
                         "query (column 3)=''", "known-query", "left merge/explode"):
            self.assertIn(expected, message)
        self.assertNotIn("12345", message)  # Never dump embedding values.
        model.assert_not_called()
        self.assertFalse((self.root / "out" / co.OUTPUT_FILENAME).exists())

    def test_legacy_assembly_can_preserve_null_query(self):
        mapping = {7: [(float("nan"), "A", "retained")]}
        legacy = self.legacy_occurrences(mapping)
        self.assertEqual(len(legacy), 1)
        name, rep, query = next(iter(legacy))
        self.assertTrue(np.isnan(query))
        self.hashes(mapping)
        # pandas to_csv writes this NaN as an empty field in glm_embeds.tsv.
        self.manifest([(name, rep, "")])
        with self.assertRaisesRegex(ValueError, r"fields \['query'\].*query candidates .*\[\]"):
            self.index()

    def test_missing_metadata_fields_are_named_and_escaped(self):
        for column in ("neighborhood_name", "rep_y", "query"):
            with self.subTest(column=column):
                fields = dict(neighborhood_name="retained", rep_y="A", query="q")
                fields[column] = " \t "
                self.manifest([tuple(fields.values())])
                index = ContextIndex(":memory:")
                try:
                    with self.assertRaises(ValueError) as raised:
                        index.load_manifest(self.root, False)
                    self.assertIn(f"['{column}']", str(raised.exception))
                    self.assertIn("' \\t '", str(raised.exception))
                finally:
                    index.close()

    def test_truncated_reference_reports_width_and_absent_field(self):
        (self.root / "glm_embeds.tsv").write_text("neighborhood_name\trep\tquery\nretained\tA\n")
        with self.assertRaisesRegex(ValueError, "expected 3 fields.*found 2.*query .*<absent>"):
            self.index()

    def test_missing_and_duplicate_reference_headers(self):
        for header, expected in (("neighborhood_name\trep\n", "missing required metadata columns.*query"),
                                 ("neighborhood_name\trep\tquery\tquery\n", "duplicate metadata columns.*query")):
            with self.subTest(header=header):
                (self.root / "glm_embeds.tsv").write_text(header)
                index = ContextIndex(":memory:")
                try:
                    with self.assertRaisesRegex(ValueError, expected):
                        index.load_manifest(self.root, False)
                finally:
                    index.close()

    def test_legacy_reference_without_rep(self):
        (self.root / "glm_embeds.tsv").write_text("query\t0\tneighborhood_name\nq\t1.0\tretained\n")
        self.assertEqual(tuple(self.index().db.execute("SELECT name,rep,query FROM targets").fetchone()),
                         ("retained", "A", "q"))

    def test_discovery_and_invalid_explicit_directory(self):
        self.manifest([("retained", "A", "q")])
        self.assertEqual(find_neighborhood_run(self.root / "masked_embeds" / "masked_embs_layer9"), self.root.resolve())
        with self.assertRaisesRegex(ValueError, "not a directory"):
            find_neighborhood_run(self.root, self.root / "missing")

    def test_signature_distinguishes_order_strand_position_and_attention(self):
        row = co.Neighborhood("a", 1, "n", ("+A", "-B", "+C"), 0, "A")
        inputs, attention = co.prepare_batch([row], self.embeddings, self.mean, self.std)
        original = co.context_signature(inputs[0], attention[0], 0)
        changed = inputs[0].clone(); changed[1, -1] *= -1
        self.assertNotEqual(original, co.context_signature(changed, attention[0], 0))
        changed = inputs[0].clone(); changed[[1, 2]] = changed[[2, 1]]
        self.assertNotEqual(original, co.context_signature(changed, attention[0], 0))
        self.assertNotEqual(original, co.context_signature(inputs[0], attention[0], 1))
        mask = attention[0].clone(); mask[2] = 0
        self.assertNotEqual(original, co.context_signature(inputs[0], mask, 0))

    def test_uncached_equal_contexts_with_tolerance_and_batch_reordering(self):
        from transformers import RobertaConfig
        from meetneighbors.predictvfs.glm.gLM import gLM

        torch.manual_seed(32)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        model = gLM(RobertaConfig(vocab_size=20, hidden_size=20, num_attention_heads=2,
                    intermediate_size=40, num_hidden_layers=9, emb_dim=1281, num_pc=4,
                    num_pred=4, predict_probs=True, tie_word_embeddings=False,
                    position_embedding_type="relative_key_query")).eval()
        rows = [co.Neighborhood("a", 1, "one", ("+A", "-B", "+C"), 0, "A"),
                co.Neighborhood("b", 1, "two", ("+D", "-B", "+C"), 0, "D")]
        inputs, attention = co.prepare_batch(rows, self.embeddings, self.mean, self.std)
        with patch.object(co, "ESM_DIM", 20):
            alone = co.embed_centers(model, inputs[:1], attention[:1], rows[:1], torch.device("cpu"))[0]
            reverse = co.embed_centers(model, inputs.flip(0), attention.flip(0), rows[::-1], torch.device("cpu"))
        for vector in reverse:
            np.testing.assert_allclose(alone, vector, rtol=1e-3, atol=5e-3)
            cosine = np.dot(alone, vector) / (np.linalg.norm(alone) * np.linalg.norm(vector))
            self.assertGreaterEqual(float(cosine), 0.99999)


if __name__ == "__main__":
    unittest.main()
