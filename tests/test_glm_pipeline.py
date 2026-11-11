"""Small, self-contained extraction and occurrence-alignment regressions."""
import logging
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dask
import numpy as np
import pandas as pd
import torch

from meetneighbors import embedding_pipeline as ep
from meetneighbors import glm_input_frm_neighbors as gi
from meetneighbors import plot_map_neighborhood_res as pn
from meetneighbors.main import get_parser, workflow
from meetneighbors.predictvfs import neighborhood_classification as nc
from meetneighbors.predictvfs.glm import glm_embed as ge


def neighborhood(name, center, strands=('+', '-', '+'), queries=None):
    records = []
    for position, (rep, start, strand) in enumerate(zip(('A', 'B', 'A'), ('2', '10', '100'), strands)):
        records.append(dict(neighborhood_name=name, rep=rep, start=start, strand=strand,
                            prot_gffname=f'p{position}!!!g', VF_center=f'p{center}', gff='g',
                            query=(queries or [f'q_{name}'])[0] if position == center else None))
    for query in (queries or [])[1:]:
        records.append({**records[center], 'query': query})
    return pd.DataFrame(records)


class GLMPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'glm_inputs'
        self.inputs.mkdir()
        self.logger = logging.getLogger('glm-tests')
        self.args = SimpleNamespace(out=str(self.root) + '/')
        torch.set_num_threads(1)

    def inputs_for(self, clusters):
        for name in sorted(clusters.neighborhood_name.unique()):
            gi.get_glm_input(query=name, uniq_neighborhoods_d={name: {name}},
                             mmseqs_clust=clusters, glm_input_dir='glm_inputs',
                             logger=self.logger, args=self.args)
        return ep.read_neighborhoods(self.inputs, clusters)

    def vectors(self, rows):
        result = np.zeros((len(rows), 5, 1280), dtype=np.float32)
        for i, row in enumerate(rows):
            for position in range(len(row.proteins)):
                result[i, position] = 100 * (ord(row.neighborhood_name[0]) - ord('a')) + position
        return result

    def test_representative_order_hash_restores_numeric_positions_and_labels(self):
        full = pd.concat([neighborhood('a', 0, queries=['q1', 'q2']),
                          neighborhood('b', 2), neighborhood('c', 1, ('+', '+', '+'))], ignore_index=True)
        full = pd.concat([full, full.iloc[[0]]], ignore_index=True).sample(frac=1, random_state=5)
        clusters, mapping = pn.hash_neighborhoods(full, self.args)
        # Historical hashing intentionally uses ordered representatives only.
        self.assertEqual(clusters.neighborhood_name.nunique(), 1)
        self.assertEqual(set(mapping), {hash(('A', 'B', 'A'))})
        rows = self.inputs_for(clusters)
        self.assertEqual(tuple(p[1:] for p in rows[0].proteins), ('A', 'B', 'A'))
        targets = ep.center_targets(rows, clusters, mapping)
        frame = ep.center_dataframe(rows, self.vectors(rows), targets)
        actual = {(r.query, r.neighborhood_name): (r.nn_center_id, frame.iloc[i][0])
                  for i, r in enumerate(frame.itertuples())}
        base = 100 * (ord(rows[0].neighborhood_name) - ord('a'))
        self.assertEqual(actual, {('q1', 'a'): (0, base), ('q2', 'a'): (0, base),
                                  ('q_b', 'b'): (2, base + 2), ('q_c', 'c'): (1, base + 1)})
        self.assertTrue(all(isinstance(r.proteins, tuple) for r in rows))
        self.assertEqual(list(frame.columns), ep.METADATA + ep.DIMENSIONS)

    def test_conflicting_occurrence_and_bad_index_rejected(self):
        bad = neighborhood('a', 0)
        bad = pd.concat([bad, bad.iloc[[0]].assign(rep='X')])
        with self.assertRaisesRegex(ValueError, 'Conflicting protein'):
            pn.hash_neighborhoods(bad, self.args)
        clusters, mapping = pn.hash_neighborhoods(neighborhood('a', 0), self.args)
        rows = self.inputs_for(clusters)
        next(iter(mapping.values()))[0][3] = 1
        with self.assertRaisesRegex(ValueError, 'does not match'):
            ep.center_targets(rows, clusters, mapping)

    def test_unhashed_labels_center_and_input_coverage(self):
        clusters = neighborhood('a', 2, queries=['q1', 'q2'])
        rows = self.inputs_for(clusters)
        self.assertEqual(rows[0].center_id, 'A')
        self.assertEqual(rows[0].center_index, 2)
        frame = ep.center_dataframe(rows, self.vectors(rows), ep.center_targets(rows, clusters, None))
        self.assertEqual(frame[0].tolist(), [2, 2])
        self.assertEqual(set(frame['query']), {'q1', 'q2'})
        with self.assertRaisesRegex(ValueError, 'do not cover'):
            ep.read_neighborhoods(self.inputs, pd.concat([clusters, neighborhood('b', 0)]))

    def test_batches_resume_corruption_partial_batch_and_identity(self):
        clusters = pd.concat([neighborhood(c, i) for i, c in enumerate('abc')])
        rows = self.inputs_for(clusters)
        out = self.root / 'glm_outputs'
        calls = []
        def interrupted(batch):
            calls.append(len(batch))
            if len(calls) == 2:
                raise RuntimeError('interrupted')
            return self.vectors(batch)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            ep.run_embedding_batches(rows, clusters, None, out, interrupted, 2, 5)
        self.assertEqual(calls, [2, 1])
        with patch.object(self, 'vectors', wraps=self.vectors) as embed:
            result = ep.run_embedding_batches(rows, clusters, None, out, embed, 2, 5, resume=True)
            self.assertEqual(embed.call_count, 1)
            self.assertEqual(len(embed.call_args.args[0]), 1)
        frame = pd.read_csv(result, sep='\t')
        self.assertEqual(frame['0'].tolist(), [0, 101, 202])
        with patch.object(self, 'vectors', side_effect=AssertionError('should reuse')):
            ep.run_embedding_batches(rows, clusters, None, out, self.vectors, 2, 5, resume=True)
        (out / 'batch_000001.glm_embeds.tsv').write_text('partial')
        with patch.object(self, 'vectors', wraps=self.vectors) as embed:
            ep.run_embedding_batches(rows, clusters, None, out, embed, 2, 5, resume=True)
            self.assertEqual(embed.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            ep.run_embedding_batches(rows, clusters, None, out, self.vectors, 1, 5, resume=True)

    def test_legacy_resume_warns_and_uses_compute_umap_first_occurrence(self):
        clusters = neighborhood('a', 2).assign(nn_hashes=123)
        rows = self.inputs_for(clusters)
        mapping = {123: [['old', 'A', 'restored']]}
        from meetneighbors import compute_umap as cu
        with self.assertLogs(level='WARNING') as logs, patch.object(
                cu, 'legacy_center_targets', wraps=cu.legacy_center_targets) as legacy:
            path = ep.run_embedding_batches(rows, clusters, mapping, self.root / 'glm_outputs',
                                            self.vectors, 2, 5, resume=True)
        legacy.assert_called_once()
        self.assertIn('Legacy hash mapping', logs.output[0])
        frame = pd.read_csv(path, sep='\t')
        self.assertEqual(frame['0'].tolist(), [0])
        self.assertEqual(frame.nn_center_id.tolist(), [0])
        self.assertEqual(frame.neighborhood_name.tolist(), ['restored'])

    def test_prepare_batch_padding_normalization_and_errors(self):
        rows = [gi.Neighborhood('a.tsv', 'a', ('q',), ('+A', '-B'), 1, 'B')]
        reps = {'A': [torch.full((1280,), 7.), 'A'], 'B': [np.full(1280, 11.), 'B']}
        inputs, attention = nc.prepare_batch(rows, reps, np.full(1280, 3), np.full(1280, 2), 4)
        np.testing.assert_array_equal(inputs[0, 0, :1280], 2)
        np.testing.assert_array_equal(inputs[0, 1, :1280], 4)
        np.testing.assert_array_equal(inputs[0, 2:], 0)
        self.assertEqual(inputs[0, 1, -1], -0.5)
        self.assertEqual(attention.tolist(), [[1, 1, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'missing neighbor'):
            nc.prepare_batch(rows, {}, np.zeros(1280), np.ones(1280), 4)

    def test_small_glm_layer_three_parity_bypasses_prediction_heads(self):
        from transformers import RobertaConfig
        from meetneighbors.predictvfs.glm.gLM import gLM
        config = RobertaConfig(vocab_size=20, hidden_size=20, num_attention_heads=2,
                               intermediate_size=40, num_hidden_layers=4,
                               tie_word_embeddings=False, emb_dim=1281, num_pc=4,
                               num_pred=4, predict_probs=True, position_embedding_type='relative_key_query')
        model = gLM(config).eval()
        inputs = torch.randn(2, 4, 1281)
        attention = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        with torch.inference_mode():
            expected = model(inputs_embeds=inputs, attention_mask=attention,
                             labels=torch.zeros(2, 4, 4), masked_tokens=attention.bool().unsqueeze(-1),
                             output_hidden_states=True).all_hidden_states[:, 3].numpy()
        with patch.object(model, 'forward', side_effect=AssertionError('full forward')), \
             patch.object(model.roberta.lm_head, 'forward', side_effect=AssertionError('prediction head')):
            actual = ge.embed_centers(model, inputs, attention, [None, None], torch.device('cpu'), 20, 3)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertEqual(actual.shape, (2, 4, 20))
        config.num_hidden_layers = 3
        short = ge.EmbeddingBackbone(config).eval()
        short.load_state_dict({key: model.state_dict()[key] for key in short.state_dict()})
        truncated = ge.embed_centers(short, inputs, attention, [None, None], torch.device('cpu'), 20, 3)
        np.testing.assert_allclose(truncated, expected, atol=1e-6)

    def test_classifier_selects_dimensions_by_name(self):
        frame = pd.DataFrame(np.ones((2, 1280)), columns=[str(i) for i in range(1280)])
        frame['query'], frame['neighborhood_name'], frame['nn_center_id'] = ['q1', 'q2'], ['a', 'b'], [99, 98]
        frame = frame[list(reversed(frame.columns))]
        class Classifier(torch.nn.Module):
            def load_state_dict(self, state):
                pass
            def forward(self, values):
                torch.testing.assert_close(values, torch.ones_like(values))
                return values[:, :2]
        with patch.object(nc.clf, 'FullyConnectedNN', return_value=Classifier()), \
             patch.object(torch, 'load', return_value={}), patch.object(nc, 'DEVICE', torch.device('cpu')):
            output = nc.get_embed_preds(frame, 'unused', SimpleNamespace(classes_=['x', 'y']), SimpleNamespace(glm_bs=1))
        self.assertEqual(output['query'].tolist(), ['q1', 'q2'])
        np.testing.assert_allclose(output[['x', 'y']], 0.5)

    def test_context_mapping_uses_indexed_repeated_occurrence(self):
        from meetneighbors.context_mapping import ContextIndex
        from meetneighbors.context_only import iter_neighborhoods
        path = self.inputs / 'a.tsv'
        path.write_text('a\t+A;-B;+A\t0\n')
        mapping_path = self.root / 'nnhash_mapping.obj'
        with mapping_path.open('wb') as handle:
            pickle.dump({'hash': [['q1', 'A', 'a', 0], ['q2', 'A', 'b', 2]]}, handle)
        index = ContextIndex(self.root / 'index.sqlite')
        self.addCleanup(index.close)
        index.add_inputs(iter_neighborhoods([path], 'original'), 'original')
        index.load_hashes(mapping_path)
        index.load_manifest(self.root, True)
        index.resolve()
        result = index.db.execute('SELECT t.name,j.center FROM targets t JOIN jobs j ON t.job_id=j.id ORDER BY t.name').fetchall()
        self.assertEqual([tuple(r) for r in result], [('a', 0), ('b', 2)])

    def write_genome(self):
        genomes = self.root / 'genomes'
        genomes.mkdir()
        (genomes / 'g.faa').write_text('>A\nMALWMRLLPLLALLALWGPDPAAA\n>B\nMKKLALALALALLLAAAGGPPVV\n>C\nMTTGGVVAAALLLPPPVVVKKLL\n')
        (genomes / 'g.gff').write_text('##gff-version 3\n' + ''.join(
            f'ctg\ttest\tCDS\t{start}\t{start + 65}\t.\t+\t0\tID={protein};protein_id={protein};locus_tag={protein}\n'
            for protein, start in [('A', 2), ('B', 100), ('C', 200)]))
        return genomes

    def test_extraction_cli_pairs_and_genomes_hash_on_off(self):
        genomes = self.write_genome()
        pairs = self.root / 'pairs.tsv'
        pairs.write_text('B\tg\nC\tg\n')
        checkpoint = self.root / 'fake-checkpoint'
        checkpoint.write_text('unit-test')
        norms = {'norm.pkl': {'mean': np.zeros(1280), 'std': np.ones(1280)},
                 'labelbinarizer_vfcategories.obj': SimpleNamespace(classes_=['x', 'y'])}
        for pairs_mode in (False, True):
            for hashed in (False, True):
                with self.subTest(pairs=pairs_mode, hashed=hashed):
                    out = self.root / f'out-{pairs_mode}-{hashed}'
                    argv = ['meetneighbors', 'extract_neighbors', '--genomes', str(genomes),
                            '--out', str(out), '--threads', '1', '--mem', '2', '--glm_bs', '1',
                            '--min_prots', '3', '--max_prots', '5', '--head_on']
                    if pairs_mode:
                        argv += ['--prot_genome_pairs', str(pairs)]
                    if hashed:
                        argv += ['--memory_optimize']
                    def embed(reps, model, rows, mean, std, maximum, layer):
                        self.assertEqual(layer, 3)
                        return np.broadcast_to(np.arange(maximum)[None, :, None], (len(rows), maximum, 1280)).copy()
                    with patch('sys.argv', argv), dask.config.set(scheduler='synchronous'), \
                         patch('meetneighbors.main.get_logger', return_value=self.logger), \
                         patch.object(nc, 'get_plm_embeds', return_value={'unused': 1}), \
                         patch.object(ge, 'load_model', return_value=object()) as load, \
                         patch.object(nc, 'embed_sequences', side_effect=embed), \
                         patch('meetneighbors.predictvfs.loader.load_pickle', return_value=norms), \
                         patch('meetneighbors.predictvfs.loader.get_glm_model_path', return_value=checkpoint):
                        workflow(get_parser())
                        load.assert_called_once()
                    frame = pd.read_csv(out / 'glm_embeds.tsv', sep='\t')
                    self.assertEqual(len(frame), 2 if pairs_mode else 3)
                    np.testing.assert_array_equal(frame['0'], frame.nn_center_id)
                    self.assertFalse(frame[ep.METADATA].isna().any().any())
                    self.assertTrue(all(frame.glminput_tsv.str.startswith('combined_chunk_')))
                    with patch('sys.argv', argv + ['--resume']), \
                         patch('meetneighbors.main.get_logger', return_value=self.logger), \
                         patch.object(nc, 'get_plm_embeds', side_effect=AssertionError('resume recomputed')), \
                         patch('meetneighbors.predictvfs.loader.load_pickle', return_value=norms), \
                         patch('meetneighbors.predictvfs.loader.get_glm_model_path', return_value=checkpoint):
                        workflow(get_parser())

    def test_query_search_multilabel_and_predictvf_resume(self):
        from meetneighbors import ring_mmseqs as mm
        genomes = self.write_genome()
        out = self.root / 'query-output'
        checkpoint = self.root / 'fake-checkpoint'
        checkpoint.write_text('unit-test')
        norms = {'norm.pkl': {'mean': np.zeros(1280), 'std': np.ones(1280)},
                 'labelbinarizer_vfcategories.obj': SimpleNamespace(classes_=['x', 'y'])}
        argv = ['meetneighbors', 'extract_neighbors', '--genomes', str(genomes),
                '--query_fasta', str(genomes / 'g.faa'), '--out', str(out), '--threads', '1',
                '--mem', '2', '--glm_bs', '1', '--min_prots', '3', '--max_prots', '5', '--memory_optimize']
        def search(args, db):
            pd.DataFrame({'query': ['Q1', 'Q2', 'Q3', 'Q1'], 'target': ['B', 'B', 'C', 'B'],
                          'tset': ['g.faa'] * 4, 'tsetid': [0] * 4}).to_csv(
                              out / 'vfs_in_genomes.tsv', sep='\t', index=False)
        def embed(reps, model, rows, mean, std, maximum, layer):
            return np.broadcast_to(np.arange(maximum)[None, :, None], (len(rows), maximum, 1280)).copy()
        with patch('sys.argv', argv), dask.config.set(scheduler='synchronous'), \
             patch('meetneighbors.main.get_logger', return_value=self.logger), \
             patch.object(mm, 'mmseqs_createdb'), patch.object(mm, 'mmseqs_search', side_effect=search), \
             patch.object(nc, 'get_plm_embeds', return_value={'unused': 1}), \
             patch.object(ge, 'load_model', return_value=object()), \
             patch.object(nc, 'embed_sequences', side_effect=embed), \
             patch('meetneighbors.predictvfs.loader.load_pickle', return_value=norms), \
             patch('meetneighbors.predictvfs.loader.get_glm_model_path', return_value=checkpoint):
            workflow(get_parser())
        frame = pd.read_csv(out / 'glm_embeds.tsv', sep='\t')
        self.assertEqual(dict(zip(frame['query'], frame.nn_center_id)), {'Q1': 1, 'Q2': 1, 'Q3': 2})
        classifier = torch.nn.Linear(1280, 2)
        with patch('sys.argv', [argv[0], 'predictvf', *argv[2:], '--resume']), \
             patch('meetneighbors.main.get_logger', return_value=self.logger), \
             patch.object(nc, 'get_plm_embeds', side_effect=AssertionError('resume recomputed')), \
             patch('meetneighbors.predictvfs.loader.load_pickle', return_value=norms), \
             patch('meetneighbors.predictvfs.loader.get_glm_model_path', return_value=checkpoint), \
             patch('meetneighbors.predictvfs.loader.load_clf_models', return_value={'nn_clf': 'unused'}), \
             patch.object(nc.clf, 'FullyConnectedNN', return_value=classifier), \
             patch.object(torch, 'load', return_value=classifier.state_dict()), \
             patch.object(nc, 'DEVICE', torch.device('cpu')):
            workflow(get_parser())
        preds = pd.read_csv(out / 'neighborhood_based_predictions.tsv', sep='\t')
        self.assertEqual(preds['query'].tolist(), frame['query'].tolist())
        np.testing.assert_allclose(preds[['x', 'y']].sum(axis=1), 1, atol=1e-6)
        from pipeline_checks import check_context_output
        check_context_output(self, out)

    def test_mmseqs_search_preserves_first_hit_with_and_without_header(self):
        from meetneighbors import neighbors_frm_mmseqs as neighbors
        columns = ['query', 'target', 'evalue', 'pident', 'qcov', 'fident',
                   'alnlen', 'qheader', 'theader', 'tset', 'tsetid']
        expected = pd.DataFrame([
            ['NR2335_000081', 'NR2335_000081', 0., 100., 1., 1., 456, 'first', 'first', 'NR2335.fasta', 0],
            ['NR2335_000082', 'NR2335_000082', 0., 100., 1., 1., 259, 'second', 'second', 'NR2335.fasta', 0],
        ], columns=columns)
        for header in (False, True):
            with self.subTest(header=header), dask.config.set(scheduler='synchronous'):
                path = self.root / 'search.tsv'
                expected.to_csv(path, sep='\t', index=False, header=header)
                groups, actual = neighbors.read_search_tsv(input_mmseqs=str(path), threads=1)
                self.assertEqual(actual['query'].tolist(), expected['query'].tolist())
                self.assertEqual(actual['target'].tolist(), expected['target'].tolist())
                self.assertEqual(len(groups.compute()[0][1]), 2)

    def test_indexed_mapping_with_legacy_embedding_files(self):
        from meetneighbors import compute_umap as cu
        clusters = neighborhood('a', 0).assign(nn_hashes=7)
        self.inputs_for(clusters)
        results = self.root / 'old_outputs' / 'a' / 'results' / 'results'
        results.mkdir(parents=True)
        with (results / 'batch.pkl.glm.embs.pkl').open('wb') as handle:
            pickle.dump([(rep, np.full(1280, position)) for position, rep in enumerate(['A', 'B', 'A'])], handle)
        frames = cu.unpack_embeddings(str(self.root / 'old_outputs'), str(self.inputs), clusters,
                                      self.logger, mem_optimize={7: [['q', 'A', 'restored', 2]]})
        frame = cu.get_glm_embeddf(frames, mem_optimize=True)
        self.assertEqual(frame[0].tolist(), [2])
        self.assertEqual(frame.nn_center_id.tolist(), [2])

    def test_umap_reads_only_completed_batches_and_numeric_features(self):
        import sys
        from meetneighbors import compute_umap as cu
        rows_df = neighborhood('a', 1)
        rows = self.inputs_for(rows_df)
        out = self.root / 'glm_outputs'
        ep.run_embedding_batches(rows, rows_df, None, out, self.vectors, 1, 5)
        (out / 'batch_999999.glm_embeds.tsv').write_text('stale')
        frames = cu.unpack_embeddings(str(out), str(self.inputs), rows_df)
        frame = cu.get_glm_embeddf(frames)
        self.assertEqual(len(frame), 1)
        class UMAP:
            def fit(self, values):
                np.testing.assert_array_equal(values, np.ones((1, 1280)))
                return self
        with patch.dict(sys.modules, {'umap': SimpleNamespace(UMAP=UMAP)}):
            cu.get_umapdf(frame)
        (out / 'batch_000000.glm_embeds.tsv').write_text('partial')
        with self.assertRaisesRegex(ValueError, 'incomplete or corrupt'):
            cu.unpack_embeddings(str(out), str(self.inputs), rows_df)


@unittest.skipUnless(os.environ.get('MEETNEIGHBORS_TEST_REAL_GLM') == '1', 'opt-in local checkpoint smoke test')
class RealCheckpointTests(unittest.TestCase):
    def test_single_cpu_batch(self):
        torch.set_num_threads(1)
        model = ge.load_model(torch.device('cpu'))
        row = gi.Neighborhood('smoke.tsv', 'smoke', ('q',), ('+A', '-B'), 1, 'B')
        reps = {'A': [np.zeros(1280), 'A'], 'B': [np.ones(1280), 'B']}
        self.assertFalse(hasattr(model.roberta, 'lm_head'))
        self.assertEqual(len(model.roberta.roberta.encoder.layer), 3)
        with patch.object(nc, 'DEVICE', torch.device('cpu')):
            actual = nc.embed_sequences(reps, model, [row], np.zeros(1280), np.ones(1280), 3, 3)
        self.assertEqual(actual.shape, (1, 3, 1280))
        self.assertTrue(np.isfinite(actual).all())
        print('REAL_CHECKPOINT_SMOKE: shape=(1, 3, 1280), all finite, prediction heads bypassed', flush=True)


if __name__ == '__main__':
    unittest.main()
