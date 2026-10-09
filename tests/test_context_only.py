"""Self-contained tests; run with python -m unittest discover -s tests -p test_context_only.py."""

import csv
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from meetneighbors import context_only as co
from meetneighbors.main import get_parser, workflow


class ContextOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "glm_inputs").mkdir()
        (self.root / "glm_outputs").mkdir()
        self.embeddings = {"A": np.full(1280, 7, dtype=np.float16),
                           "B": np.full(1280, 11, dtype=np.float16)}
        self.mean = np.full(1280, 3, dtype=np.float32)
        self.std = np.full(1280, 2, dtype=np.float32)

    def write_tsv(self, text, name="a.tsv"):
        path = self.root / "glm_inputs" / name
        path.write_text(text)
        return path

    def write_esm(self, entries=None):
        path = self.root / "glm_outputs" / co.ESM_FILENAME
        with path.open("wb") as handle:
            pickle.dump(list(self.embeddings.items()) if entries is None else entries, handle)
        return path

    def test_mask_before_normalization_and_only_at_center(self):
        path = self.write_tsv("n1\t+A_masked;-A;+B\t0\nn2\t-A;+B_masked\t1\n")
        rows = list(co.iter_neighborhoods([path], "masked"))
        inputs, attention = co.prepare_batch(rows, self.embeddings, self.mean, self.std)
        np.testing.assert_array_equal(inputs[0, 0, :1280], -2)  # (-1 - 3) / 2
        np.testing.assert_array_equal(inputs[0, 1, :1280], 2)  # A remains intact as neighbor
        np.testing.assert_array_equal(inputs[1, 1, :1280], -2)
        self.assertEqual(inputs[0, 0, -1], 0.5)
        self.assertEqual(inputs[0, 1, -1], -0.5)
        np.testing.assert_array_equal(inputs[0, 3:], 0)
        self.assertEqual(attention.sum().item(), 5)
        np.testing.assert_array_equal(self.embeddings["A"], 7)
        self.assertEqual([r.center_id for r in rows], ["A", "B"])

    def test_original_and_masked_inputs_match(self):
        original = self.write_tsv("n\t-B;+A\t0\n", "original.tsv")
        masked = self.write_tsv("n\t-B_masked;+A\t0\n", "masked.tsv")
        original_rows = list(co.iter_neighborhoods([original], "original"))
        masked_rows = list(co.iter_neighborhoods([masked], "masked"))
        first = co.prepare_batch(original_rows, self.embeddings, self.mean, self.std)
        second = co.prepare_batch(masked_rows, self.embeddings, self.mean, self.std)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        self.assertEqual(original_rows[0].center_id, masked_rows[0].center_id)

    def test_remove_only_one_terminal_suffix(self):
        path = self.write_tsv("n\t+A_masked_middle_masked_masked\t0\n")
        row = next(co.iter_neighborhoods([path], "masked"))
        self.assertEqual(row.center_id, "A_masked_middle_masked")
        row = next(co.iter_neighborhoods([path], "original"))
        self.assertEqual(row.center_id, "A_masked_middle_masked_masked")

    def test_invalid_rows_have_location(self):
        for text in ["n\t+A\t-1\n", "n\t+A\t1\n", "n\t+A\t0.5\n",
                     "n\tA\t0\n", "n\t+\t0\n", "n\t+A\n",
                     "n\t" + ";".join(["+A"] * 31) + "\t0\n"]:
            with self.subTest(text=text):
                path = self.write_tsv(text)
                with self.assertRaisesRegex(ValueError, "a.tsv:1"):
                    list(co.iter_neighborhoods([path], "original"))
        path = self.write_tsv("n\t+A\t0\n")
        with self.assertRaisesRegex(ValueError, "masked mode"):
            list(co.iter_neighborhoods([path], "masked"))

    def test_missing_neighbor_has_location(self):
        path = self.write_tsv("n\t+A_masked;+missing\t0\n")
        rows = list(co.iter_neighborhoods([path], "masked"))
        with self.assertRaisesRegex(ValueError, "a.tsv:1: missing neighbor.*missing"):
            co.prepare_batch(rows, self.embeddings, self.mean, self.std)

    def test_bad_esm_vectors(self):
        for value in [np.ones(1279), np.full(1280, np.nan)]:
            with self.subTest(shape=value.shape):
                path = self.write_esm([("A", value)])
                with self.assertRaisesRegex(ValueError, "1280 finite"):
                    co.load_esm_embeddings(path)

    def test_layer_nine_projection_and_center_selection(self):
        from transformers import RobertaConfig
        from meetneighbors.predictvfs.glm.gLM import gLM

        torch.manual_seed(12)
        config = RobertaConfig(vocab_size=20, hidden_size=20, num_attention_heads=2,
                               intermediate_size=40, num_hidden_layers=10,
                               tie_word_embeddings=False,
                               emb_dim=1281, num_pc=4, num_pred=4, predict_probs=True,
                               position_embedding_type="relative_key_query")
        model = gLM(config).eval()
        path = self.write_tsv("n1\t+A;-B\t0\nn2\t+A;-B\t1\n")
        rows = list(co.iter_neighborhoods([path], "original"))
        inputs, attention = co.prepare_batch(rows, self.embeddings, self.mean, self.std)
        with torch.inference_mode():
            states = model.roberta.roberta(inputs_embeds=model.dense(inputs),
                                           attention_mask=attention,
                                           output_hidden_states=True).hidden_states
        with patch.object(co, "ESM_DIM", 20):
            actual = co.embed_centers(model, inputs, attention, rows, torch.device("cpu"))
        np.testing.assert_allclose(actual, states[9][[0, 1], [0, 1]].numpy(), atol=1e-6)
        self.assertFalse(np.allclose(actual, states[10][[0, 1], [0, 1]].numpy()))

    def test_output_order_partial_batch_and_collision(self):
        self.write_tsv("not a neighborhood TSV", "._a.tsv")
        self.write_tsv("last\t+B;+A_masked\t1\n", "z.tsv")
        self.write_tsv("first\t+A_masked;+B\t0\nsecond\t+A;-B_masked\t1\n")
        self.write_esm()
        out = self.root / "out"
        calls = []

        def fake_embed(model, inputs, attention, rows, device):
            calls.append(len(rows))
            return np.stack([np.full(1280, r.center_index) for r in rows])

        with patch.object(co, "load_normalization", return_value=(self.mean, self.std)), \
                patch.object(co, "load_model") as load, \
                patch.object(co, "embed_centers", side_effect=fake_embed):
            result = co.run_context_only(self.root, out, batch_size=2, device="cpu")
            load.assert_called_once()
            with result.open() as handle:
                table = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(calls, [2, 1])
            self.assertEqual(len(table[0]), 1284)
            self.assertEqual([row[:4] for row in table[1:]], [
                ["a.tsv", "1", "first", "A"], ["a.tsv", "2", "second", "B"],
                ["z.tsv", "1", "last", "A"]])
            self.assertEqual([float(row[4]) for row in table[1:]], [0, 1, 1])
            with self.assertRaises(FileExistsError):
                co.run_context_only(self.root, out)

    def test_no_inputs_and_preflight_failure(self):
        with self.assertRaisesRegex(ValueError, "No input TSVs"):
            co.run_context_only(self.root, self.root / "out")
        self.write_tsv("")
        self.write_esm()
        with patch.object(co, "load_normalization", return_value=(self.mean, self.std)), \
                patch.object(co, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "no neighborhoods"):
                co.run_context_only(self.root, self.root / "out", device="cpu")
            self.write_tsv("bad\t+A;+absent\t0\n")
            with self.assertRaisesRegex(ValueError, "missing neighbor"):
                co.run_context_only(self.root, self.root / "out", device="cpu", input_mode="original")
            load.assert_not_called()
        self.assertFalse((self.root / "out" / co.OUTPUT_FILENAME).exists())

    def test_inference_failure_removes_partial_output(self):
        self.write_tsv("n\t+A_masked\t0\n")
        self.write_esm()
        with patch.object(co, "load_normalization", return_value=(self.mean, self.std)), \
                patch.object(co, "load_model"), \
                patch.object(co, "embed_centers", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                co.run_context_only(self.root, self.root / "out", device="cpu")
        self.assertFalse((self.root / "out" / co.OUTPUT_FILENAME).exists())

    def test_cli_help_and_dispatch_without_genome_args(self):
        parser = get_parser()
        args = ["context_only", "--input-dir", str(self.root), "-o", str(self.root / "out")]
        parsed = parser.parse_args(args)
        self.assertEqual(parsed.glm_bs, 100)
        self.assertEqual(parsed.input_mode, "masked")
        self.assertFalse(hasattr(parsed, "genomes"))
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaises(SystemExit) as exc:
                parser.parse_args(["context_only", "--help"])
            self.assertEqual(exc.exception.code, 0)
            self.assertIn("--input-dir", stdout.getvalue())
        with patch("sys.argv", ["meetneighbors", *args]), \
                patch("meetneighbors.main.get_logger") as logger, \
                patch.object(co, "run_context_only") as run:
            workflow(parser)
            run.assert_called_once_with(str(self.root), str(self.root / "out"), 100,
                                        "auto", "masked", logger=logger.return_value)

    def test_device_selection_and_invalid_batch_size(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(co.select_device("auto").type, "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                co.select_device("cuda")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            co.run_context_only(self.root, self.root / "out", batch_size=0)


if __name__ == "__main__":
    unittest.main()
