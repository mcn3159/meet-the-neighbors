"""Dependency-free tests: PYTHONPATH=src python -m unittest discover -s tests -p test_viz_neighborhoods.py."""

import csv
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from meetneighbors import viz_neighborhoods as viz
from meetneighbors.main import get_parser, workflow


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "extraction"
        self.run.mkdir()
        self.genomes = self.root / "genomes"
        self.genomes.mkdir()
        self.names = ["A!!!g1!!!ctg!!!100-480", "X!!!g2!!!ctg!!!500-980", "B!!!g1!!!ctg!!!100-480"]
        self.members = [
            [("A", 100, "+"), ("B", 180, "-"), ("C", 400, "+")],
            [("X", 800, "-"), ("Y", 700, "+"), ("Z", 500, "-")],
            [("A", 100, "+"), ("B", 180, "-"), ("C", 400, "+")],
        ]
        self.products = {"A": "DNA-binding protein; alpha + beta", "B": "No product annotation",
                         "C": '</script><script>alert("GFF")</script>', "X": "Anchor two", "Y": "Neighbor two", "Z": "Other"}
        gffs = {
            "g1": [("A", 100, 220, "+", "protein_id=extdb:A;product=DNA-binding%20protein%3B%20alpha%20%2B%20beta"),
                   ("B", 180, 300, "-", "locus_tag=B"),
                   ("C", 400, 480, "+", "protein_id=C;product=%3C/script%3E%3Cscript%3Ealert(%22GFF%22)%3C/script%3E")],
            "g2": [("X", 800, 980, "-", "protein_id=X;product=Anchor two"),
                   ("Y", 700, 820, "+", "protein_id=Y;product=Neighbor two"),
                   ("Z", 500, 600, "-", "protein_id=Z;product=Other")],
        }
        for genome, records in gffs.items():
            (self.genomes / f"{genome}.gff").write_text("##gff-version 3\n" + "".join(
                f"ctg\ttest\tCDS\t{start}\t{end}\t.\t{strand}\t0\t{attrs}\n"
                for protein, start, end, strand, attrs in records))
            sequences = []
            for protein, *_ in records:
                family = {"X": "A", "Y": "B"}.get(protein, protein)
                sequence = "M" + "".join(random.Random(family).choices("ACDEFGHIKLMNPQRSTVWY", k=150))
                sequences.append(f">{protein}\n{sequence}\n")
            (self.genomes / f"{genome}.faa").write_text("".join(sequences))
        self.write_membership()

    def write_membership(self):
        with (self.run / "neighborhoods_db.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("locus_tag", "start", "strand", "neighborhood_name"))
            for name, members in zip(self.names, self.members):
                for protein, start, strand in members:
                    writer.writerow((protein, start, strand, name))

    def encoded(self, index, member):
        protein, start, strand = member
        return f"{protein}!!!{self.names[index]}!!!{start}!!!{strand}"

    def write_raw_clusters(self, encoded=False):
        with (self.run / "combined_fastas_clust_res.tsv").open("w") as handle:
            seen = set()
            for index, members in enumerate(self.members):
                for member in members:
                    protein = member[0]
                    rep = "family_anchor" if protein in ("A", "X") else "family_neighbor" if protein in ("B", "Y") else protein
                    identifier = self.encoded(index, member) if encoded else protein
                    if identifier not in seen:
                        handle.write(f"{rep}\t{identifier}\n")
                        seen.add(identifier)

    def render(self, names=None, **kwargs):
        out = kwargs.pop("out", self.root / "viz")
        if not ("genomes" in kwargs or "genome_tsv" in kwargs):
            kwargs["genomes"] = self.genomes
        return viz.viz_neighborhoods(self.run, out, names, **kwargs)

    def data(self, path):
        html = path.read_text()
        payload = html.split('<script id="neighborhood-data" type="application/json">', 1)[1].split("</script>", 1)[0]
        return json.loads(payload)

    def test_selection_single_sequence_mixed_and_seed(self):
        self.assertEqual(viz._select(self.names, self.names[1], None, None), [self.names[1]])
        self.assertEqual(viz._select(self.names, [self.names[1], self.names[0], self.names[1]], None, 1), self.names[1::-1])
        mixed = viz._select(self.names, self.names[0], 2, 42)
        self.assertEqual(mixed[0], self.names[0])
        self.assertEqual(set(mixed), set(self.names))
        self.assertEqual(viz._select(self.names, None, 2, 42), viz._select(reversed(self.names), None, 2, 42))

    def test_selection_errors_and_exhaustion(self):
        for names, count, message in [("missing", 0, "None of the requested"), (None, -1, "nonnegative"),
                                      (None, 0, "No neighborhoods"), (None, 1.5, "nonnegative")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                viz._select(self.names, names, count, None)
        with self.assertLogs(viz.LOG, level="WARNING") as logs:
            selected = viz._select(self.names, self.names[0], 10, 42)
        self.assertEqual(len(selected), 3)
        self.assertIn("only 2 remain", logs.output[0])

    def test_missing_names_warn_and_preserve_matching_order(self):
        requested = ["absent_b", self.names[1], "absent_a", self.names[0], self.names[1]]
        with self.assertLogs(viz.LOG, level="WARNING") as logs:
            selected = viz._select(self.names, requested, None, 42)
        self.assertEqual(selected, [self.names[1], self.names[0]])
        self.assertEqual(len(logs.output), 1)
        self.assertIn("absent_a, absent_b", logs.output[0])

    def test_missing_names_preserve_seeded_random_additions(self):
        expected = viz._select(self.names, self.names[0], 1, 42)
        with self.assertLogs(viz.LOG, level="WARNING"):
            selected = viz._select(self.names, ["absent", self.names[0]], 1, 42)
        self.assertEqual(selected, expected)
        self.assertEqual(len(selected), 2)

    def test_all_requested_names_missing_errors_before_reclustering(self):
        for names, count in [("absent", None), (["absent_a", "absent_b"], 0), (["absent"], 2)]:
            with self.subTest(names=names, count=count), patch.object(viz, "_recluster") as recluster:
                with self.assertLogs(viz.LOG, level="WARNING"), self.assertRaisesRegex(ValueError, "None of the requested"):
                    self.render(names, n_random=count, recluster=True)
                recluster.assert_not_called()
                self.assertFalse((self.root / "viz").exists())

    def test_annotations_orientation_lengths_and_lanes(self):
        path = self.render(self.names[:2])
        first, second = self.data(path)["tracks"]
        self.assertEqual({g["protein"]: g["product"] for g in first["genes"]}, {k: self.products[k] for k in ("A", "B", "C")})
        for track in (first, second):
            anchor = next(g for g in track["genes"] if g["anchor"])
            self.assertEqual(anchor["direction"], 1)
            self.assertEqual(anchor["left"] + anchor["right"], 0)
            for g in track["genes"]:
                self.assertEqual(g["right"] - g["left"], g["end"] - g["start"] + 1)
            self.assertEqual(track["lanes"], 2)
        self.assertTrue(second["flipped"])
        anchor = next(g for g in second["genes"] if g["anchor"])
        self.assertEqual((anchor["start"], anchor["end"], anchor["strand"]), (800, 980, "-"))

    def test_offline_and_safe_embedding(self):
        path = self.render(self.names[0])
        html = path.read_text()
        self.assertNotIn(self.products["C"], html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('<link ', html)
        self.assertIn("textContent", html)
        self.assertNotIn("innerHTML", html)
        self.assertEqual(self.data(path)["tracks"][0]["genes"][2]["product"], self.products["C"])

    def test_saved_clusters_shared_across_distinct_tracks(self):
        self.write_raw_clusters()
        data = self.data(self.render(self.names[:2]))
        self.assertEqual(data["shared_clusters"], ["family_anchor", "family_neighbor"])
        one = self.data(self.render(self.names[0], out=self.root / "one"))
        self.assertEqual(one["shared_clusters"], [])

    def test_synthetic_self_reps_are_unclustered_even_when_overlapping(self):
        directory = self.run / "clust_res_in_neighborhoods"
        directory.mkdir()
        with (directory / "mmseqs_clust.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("locus_tag", "start", "strand", "neighborhood_name", "rep"))
            for i, members in enumerate(self.members):
                for member in members:
                    encoded = self.encoded(i, member)
                    writer.writerow((encoded, member[1], member[2], self.names[i], encoded))
        data = self.data(self.render(self.names))
        self.assertIn("No verified", data["message"])
        self.assertTrue(all(g["cluster"] is None for t in data["tracks"] for g in t["genes"]))

    def test_partition_headers_restore_reduced_neighborhoods(self):
        (self.run / "neighborhoods_db.tsv").unlink()
        with (self.run / "combined_fasta_partition0.faa").open("w") as handle:
            for i, members in enumerate(self.members):
                for member in members:
                    handle.write(f">{self.encoded(i, member)} annotation\nMAAAA\n")
        self.write_raw_clusters(encoded=True)
        directory = self.run / "clust_res_in_neighborhoods"
        directory.mkdir()
        (directory / "mmseqs_clust.tsv").write_text("locus_tag\tstart\tstrand\tneighborhood_name\trep\n" +
            f"A\t100\t+\t{self.names[0]}\tfamily_anchor\n")
        data = self.data(self.render(self.names))
        self.assertEqual([len(t["genes"]) for t in data["tracks"]], [3, 3, 3])
        self.assertEqual(data["tracks"][0]["genes"][0]["cluster"], "family_anchor")
        self.assertIsNone(data["tracks"][2]["genes"][0]["cluster"])
        self.assertIn("8 gene occurrences have no saved assignment", data["message"])

    def test_neighborhood_table_overrides_conflicting_original_clusters(self):
        self.write_raw_clusters(encoded=True)
        directory = self.run / "clust_res_in_neighborhoods"
        directory.mkdir()
        (directory / "mmseqs_clust.tsv").write_text(
            "locus_tag\tstart\tstrand\tneighborhood_name\trep\n"
            f"A\t100\t+\t{self.names[0]}\tpreferred_family\n"
            f"X\t800\t-\t{self.names[1]}\tpreferred_family\n")
        # Even contradictory assignments within the unused source are ignored.
        with (self.run / "combined_fastas_clust_res.tsv").open("a") as handle:
            handle.write(f"another_family\t{self.encoded(0, self.members[0][0])}\n")
        data = self.data(self.render(self.names[:2]))
        self.assertEqual(data["shared_clusters"], ["preferred_family"])
        for track in data["tracks"]:
            for gene in track["genes"]:
                self.assertEqual(gene["cluster"], "preferred_family" if gene["anchor"] else None)

    def test_conflicts_within_default_source_report_file_and_row(self):
        directory = self.run / "clust_res_in_neighborhoods"
        directory.mkdir()
        (directory / "mmseqs_clust.tsv").write_text(
            "locus_tag\tstart\tstrand\tneighborhood_name\trep\n"
            f"A\t100\t+\t{self.names[0]}\tfamily_one\n"
            f"A\t100\t+\t{self.names[0]}\tfamily_two\n")
        with self.assertRaisesRegex(ValueError, r"clust_res_in_neighborhoods/mmseqs_clust.tsv:3: Conflicting"):
            self.render(self.names[0])

    def test_unselected_neighborhood_conflict_does_not_block_visualization(self):
        directory = self.run / "clust_res_in_neighborhoods"
        directory.mkdir()
        (directory / "mmseqs_clust.tsv").write_text(
            "locus_tag\tstart\tstrand\tneighborhood_name\trep\n"
            f"A\t100\t+\t{self.names[0]}\tselected_family\n"
            f"X\t800\t-\t{self.names[1]}\tfamily_one\n"
            f"X\t800\t-\t{self.names[1]}\tfamily_two\n")
        with self.assertLogs(viz.LOG, level="WARNING") as logs:
            data = self.data(self.render(self.names[0]))
        self.assertEqual(data["tracks"][0]["genes"][0]["cluster"], "selected_family")
        self.assertIn("Skipped 1 conflicting", logs.output[0])
        with self.assertRaisesRegex(ValueError, "affects selected neighborhood.*use --recluster"):
            self.render(self.names[1], out=self.root / "conflicted")
        self.assertFalse((self.root / "conflicted/neighborhoods.html").exists())

    def test_unselected_raw_assignment_conflict_does_not_block_visualization(self):
        self.write_raw_clusters()
        with (self.run / "combined_fastas_clust_res.tsv").open("a") as handle:
            handle.write("other_family\tX\n")
        data = self.data(self.render(self.names[0]))
        self.assertEqual(data["tracks"][0]["genes"][0]["cluster"], "family_anchor")
        with self.assertRaisesRegex(ValueError, "Conflicting cluster assignments.*use --recluster"):
            self.render(self.names[1], out=self.root / "conflicted")

    def test_combined_fasta_fallback_and_encoded_assignments_only(self):
        (self.run / "neighborhoods_db.tsv").unlink()
        combined = self.run / "combined.fasta"
        combined.write_text("".join(f">{self.encoded(0, m)}\nMAAAA\n" for m in self.members[0]))
        self.assertEqual(len(self.data(self.render(self.names[0]))["tracks"][0]["genes"]), 3)
        combined.unlink()
        self.write_raw_clusters(encoded=True)
        self.assertEqual(len(self.data(self.render(self.names[1], out=self.root / "raw"))["tracks"]), 1)

    def test_relative_source_manifest(self):
        manifest = self.root / "sources.tsv"
        manifest.write_text("g1\tgenomes/g1.faa\tgenomes/g1.gff\ng2\tgenomes/g2.faa\tgenomes/g2.gff\n")
        data = self.data(self.render(self.names[:2], genome_tsv=manifest))
        self.assertEqual(len(data["tracks"]), 2)

    def test_gff_errors(self):
        path = self.genomes / "g1.gff"
        original = path.read_text()
        with self.subTest("missing"):
            path.write_text(original.replace("protein_id=extdb:A", "protein_id=missing"))
            with self.assertRaisesRegex(ValueError, "Missing GFF CDS match"):
                self.render(self.names[0])
        with self.subTest("ambiguous"):
            path.write_text(original + original.splitlines()[1] + "\n")
            with self.assertRaisesRegex(ValueError, "Ambiguous GFF CDS match"):
                self.render(self.names[0])

    def test_duplicate_identifiers_across_genomes_have_distinct_annotations(self):
        path = self.genomes / "g2.gff"
        path.write_text(path.read_text().replace("protein_id=Y", "protein_id=A"))
        self.members[1][1] = ("A", 700, "+")
        self.write_membership()
        data = self.data(self.render(self.names[:2]))
        second = next(g for g in data["tracks"][1]["genes"] if g["protein"] == "A")
        self.assertEqual(second["product"], "Neighbor two")
        self.write_raw_clusters()
        with self.assertRaisesRegex(ValueError, "Ambiguous bare protein ID"):
            self.render(self.names[:2], out=self.root / "ambiguous")
        fasta = self.genomes / "g2.faa"
        fasta.write_text(fasta.read_text().replace(">Y\n", ">A\n"))
        with patch.object(viz.shutil, "which", return_value="mmseqs"), patch.object(viz.subprocess, "run", side_effect=self.fake_mmseqs):
            path = self.render(self.names[:2], out=self.root / "resolved", recluster=True)
        with (path.parent / "recluster/protein_mapping.tsv").open() as handle:
            rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["protein_id"] == "A"]
        self.assertEqual(len({r["internal_id"] for r in rows}), 2)

    def test_contextual_table_without_raw_assignments(self):
        directory = self.run / "clust_res_in_neighborhoods"
        directory.mkdir()
        with (directory / "mmseqs_clust.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("locus_tag", "start", "strand", "neighborhood_name", "rep"))
            for i, members in enumerate(self.members[:2]):
                for member in members:
                    rep = "shared" if member[0] in ("A", "X") else member[0]
                    writer.writerow((self.encoded(i, member), member[1], member[2], self.names[i], rep))
        (self.run / "neighborhoods_db.tsv").unlink()
        data = self.data(self.render(self.names[:2]))
        self.assertEqual(data["shared_clusters"], ["shared"])

    def test_conflicting_assignments(self):
        self.write_raw_clusters()
        with (self.run / "combined_fastas_clust_res.tsv").open("a") as handle:
            handle.write("different\tA\n")
        with self.assertRaisesRegex(ValueError, "Conflicting cluster"):
            self.render(self.names[0])

    def test_output_and_sources_validation(self):
        path = self.render(self.names[0])
        content = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.render(self.names[0])
        self.assertEqual(path.read_bytes(), content)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            viz.viz_neighborhoods(self.run, self.root / "missing")
        with self.assertRaisesRegex(ValueError, "cluster_dir requires"):
            self.render(self.names[0], out=self.root / "other", cluster_dir=self.root / "clusters")

    def fake_mmseqs(self, command, **kwargs):
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, stdout="test-version\n")
        if command[1] == "createtsv":
            directory = Path(command[-1]).parent
            ids = [name for name, _ in viz._fasta(directory / "selected_proteins.fasta")]
            Path(command[-1]).write_text("".join(f"{ids[0]}\t{identifier}\n" for identifier in ids))
        if command[1] == "convert2fasta":
            Path(command[-1]).write_text(">protein_000001\nMAAAA\n")
        return subprocess.CompletedProcess(command, 0)

    @patch.object(viz.shutil, "which", return_value="/test path/mmseqs")
    def test_recluster_selected_sequences_mapping_and_settings(self, which):
        before = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        with patch.object(viz.subprocess, "run", side_effect=self.fake_mmseqs) as run:
            path = self.render([self.names[0], self.names[2]], recluster=True, seq_id=.5, cov=.8,
                               cov_mode=1, sensitivity=7, cluster_mode=2, threads=2, mem=3)
        directory = path.parent / "recluster"
        self.assertEqual(len(list(viz._fasta(directory / "selected_proteins.fasta"))), 3)
        with (directory / "protein_mapping.tsv").open() as handle:
            mapping = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(mapping), 6)
        self.assertEqual(len({row["internal_id"] for row in mapping}), 3)
        command = next(call.args[0] for call in run.call_args_list if call.args[0][1] == "cluster")
        for flag, value in (("--min-seq-id", "0.5"), ("-c", "0.8"), ("--cov-mode", "1"),
                            ("-s", "7"), ("--cluster-mode", "2"), ("--threads", "2"), ("--split-memory-limit", "3G")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertTrue(all(not call.kwargs.get("shell") for call in run.call_args_list))
        metadata = json.loads((directory / "run.json").read_text())
        self.assertEqual(metadata["version"], "test-version")
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(self.data(path)["shared_clusters"], ["protein_000001"])
        self.assertTrue((directory / "representatives.fasta").is_file())
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()})

    def test_partial_match_reclusters_only_matching_neighborhood_proteins(self):
        # Neither a missing name nor an unselected genome should require sources.
        (self.genomes / "g2.gff").unlink()
        (self.genomes / "g2.faa").unlink()
        missing = "unknown!!!absent_genome!!!ctg!!!1-100"
        with patch.object(viz.shutil, "which", return_value="mmseqs"), patch.object(viz.subprocess, "run", side_effect=self.fake_mmseqs):
            with self.assertLogs(viz.LOG, level="WARNING") as logs:
                path = self.render([missing, self.names[0]], recluster=True)
        self.assertIn(missing, logs.output[0])
        self.assertEqual([track["name"] for track in self.data(path)["tracks"]], [self.names[0]])
        directory = path.parent / "recluster"
        with (directory / "protein_mapping.tsv").open() as handle:
            mapping = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual({row["neighborhood_name"] for row in mapping}, {self.names[0]})
        self.assertEqual({row["protein_id"] for row in mapping}, {"A", "B", "C"})
        self.assertEqual(len(mapping), 3)
        actual_sequences = [sequence for _, sequence in viz._fasta(directory / "selected_proteins.fasta")]
        expected_sequences = [sequence for _, sequence in viz._fasta(self.genomes / "g1.faa")]
        self.assertCountEqual(actual_sequences, expected_sequences)
        self.assertEqual(json.loads((directory / "run.json").read_text())["neighborhoods"], [self.names[0]])

    def test_recluster_failures_and_no_overwrite(self):
        with patch.object(viz.shutil, "which", return_value=None), self.assertRaisesRegex(ValueError, "not found on PATH"):
            self.render(self.names[0], recluster=True)
        directory = self.root / "clusters"
        directory.mkdir()
        with self.assertRaises(FileExistsError):
            self.render(self.names[0], recluster=True, cluster_dir=directory)
        with patch.object(viz.shutil, "which", return_value="mmseqs"):
            with patch.object(viz.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["mmseqs", "cluster"])):
                with self.assertRaisesRegex(ValueError, "reclustering failed"):
                    self.render(self.names[0], recluster=True)
        self.assertFalse((self.root / "viz" / "neighborhoods.html").exists())
        self.assertEqual(json.loads((self.root / "viz/recluster/run.json").read_text())["status"], "failed")

    def test_missing_sequences_fail_before_creating_cluster_directory(self):
        (self.genomes / "g1.faa").write_text(">A\nMAAAA\n")
        with patch.object(viz.shutil, "which", return_value="mmseqs"), self.assertRaisesRegex(ValueError, "Missing protein sequences"):
            self.render(self.names[0], recluster=True)
        self.assertFalse((self.root / "viz/recluster").exists())

    def test_incomplete_recluster_assignments_do_not_publish_html(self):
        def incomplete(command, **kwargs):
            result = self.fake_mmseqs(command, **kwargs)
            if command[1] == "createtsv":
                Path(command[-1]).write_text("protein_000001\tprotein_000001\n")
            return result
        with patch.object(viz.shutil, "which", return_value="mmseqs"), patch.object(viz.subprocess, "run", side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, "incomplete assignments"):
                self.render(self.names[0], recluster=True)
        self.assertFalse((self.root / "viz/neighborhoods.html").exists())
        self.assertEqual(json.loads((self.root / "viz/recluster/run.json").read_text())["status"], "failed")

    def test_invalid_recluster_settings(self):
        for settings in ({"seq_id": 90}, {"cov": -1}, {"threads": 0}, {"mem": float("nan")},
                         {"cov_mode": 9}, {"cluster_mode": 2.5}, {"sensitivity": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.render(self.names[0], recluster=True, **settings)

    def test_cli_dispatch_does_not_import_ml_stack(self):
        argv = ["meetneighbors", "viz_neighborhoods", "--input-dir", str(self.run), "--out", str(self.root / "cli"),
                "--genomes", str(self.genomes), "--neighborhood-names", "absent", self.names[0]]
        with patch.object(sys, "argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertLogs(viz.LOG, level="WARNING") as logs:
                workflow(get_parser())
        self.assertIn("absent", logs.output[0])
        self.assertTrue((self.root / "cli/neighborhoods.html").is_file())
        self.assertIn("neighborhoods.html", stdout.getvalue())
        # Other suites legitimately import the ML workflow. Check lazy dispatch
        # in a fresh interpreter so test order cannot change this assertion.
        isolated_argv = [str(self.root / "cli_isolated") if item == str(self.root / "cli") else item
                         for item in argv]
        code = ("import sys; from meetneighbors.main import get_parser, workflow; "
                f"sys.argv = {isolated_argv!r}; workflow(get_parser()); "
                "assert 'meetneighbors.predictvfs.neighborhood_classification' not in sys.modules")
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("mmseqs"), "MMseqs2 is not installed")
    def test_real_mmseqs_smoke(self):
        try:
            path = self.render(self.names[:2], recluster=True, threads=1)
        except ValueError:
            log = self.root / "viz/recluster/mmseqs.log"
            if log.exists():
                self.fail(log.read_text()[-5000:])
            raise
        self.assertTrue(path.is_file())
        metadata = json.loads((path.parent / "recluster/run.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        data = self.data(path)
        first = {g["protein"]: g["cluster"] for g in data["tracks"][0]["genes"]}
        second = {g["protein"]: g["cluster"] for g in data["tracks"][1]["genes"]}
        self.assertEqual(first["A"], second["X"])
        self.assertEqual(first["B"], second["Y"])
        self.assertNotEqual(first["A"], first["B"])


if __name__ == "__main__":
    unittest.main()
