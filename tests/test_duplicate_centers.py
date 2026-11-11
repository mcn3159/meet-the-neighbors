"""Occurrence-level regressions for duplicated VF names, without large models."""
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from meetneighbors import embedding_pipeline as ep
from meetneighbors import glm_input_frm_neighbors as gi
from meetneighbors.predictvfs import neighborhood_classification as nc
from meetneighbors.predictvfs import clf_models
import test_glm_pipeline as fixtures


def duplicated_neighborhood(name='a'):
    frame = fixtures.neighborhood(name, 0)
    frame.loc[2, 'prot_gffname'] = frame.loc[0, 'prot_gffname']
    frame.loc[[0, 2], 'query'] = 'duplicated-query'
    frame.loc[1, 'query'] = 'neighbor-query'
    return frame


class DuplicateCenterTests(unittest.TestCase):
    inputs_for = fixtures.GLMPipelineTests.inputs_for
    vectors = fixtures.GLMPipelineTests.vectors

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'glm_inputs'
        self.inputs.mkdir()
        self.logger = logging.getLogger('duplicate-centers')
        self.args = SimpleNamespace(out=str(self.root) + '/')
        torch.set_num_threads(1)

    def test_nr2335_reported_duplicate_coordinates(self):
        # Repeated identifier/coordinates taken from tests/test_data/genomes/NR2335.gff.
        name = 'NR2335_000081!!!NR2335!!!cluster_001!!!84000-86500'
        frame = pd.DataFrame([
            dict(start=str(start), strand='+', rep='NR2335_000081',
                 prot_gffname='NR2335_000081!!!NR2335', VF_center='NR2335_000081',
                 gff='NR2335', neighborhood_name=name, query='NR2335_000081')
            for start in [84654, 85134]])
        rows = self.inputs_for(frame)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].center_index, 0)
        self.assertEqual(len(Path(rows[0].source_file).read_text().splitlines()), 1)
        calls = []
        def embed(batch):
            calls.append(len(batch))
            return np.broadcast_to(np.arange(2)[None, :, None], (len(batch), 2, 1280)).copy()
        path = ep.run_embedding_batches(rows, frame, None, self.root / 'glm_outputs', embed, 1, 2)
        output = pd.read_csv(path, sep='\t')
        self.assertEqual(calls, [1])
        self.assertEqual(output.nn_center_id.tolist(), [0, 1])
        self.assertEqual(output['0'].tolist(), [0, 1])
        self.assertEqual(output['query'].tolist(), ['NR2335_000081'] * 2)

    def test_query_labels_belong_to_occurrence_not_gene_name(self):
        frame = duplicated_neighborhood()
        frame = pd.concat([frame, frame.iloc[[0]].assign(query='first-only'),
                           frame.iloc[[2]].assign(query='second-only'), frame.iloc[[0]]])
        frame = frame.sample(frac=1, random_state=17)
        rows = self.inputs_for(frame)
        output = ep.center_dataframe(rows, self.vectors(rows), ep.center_targets(rows, frame, None))
        actual = {(r['query'], r['nn_center_id']): r[0] for _, r in output.iterrows()}
        self.assertEqual(actual, {('duplicated-query', 0): 0, ('duplicated-query', 2): 2,
                                  ('first-only', 0): 0, ('second-only', 2): 2})

    def test_reference_index_can_be_second_copy_but_not_neighbor_or_out_of_bounds(self):
        frame = duplicated_neighborhood()
        rows = self.inputs_for(frame)
        path = Path(rows[0].source_file)
        text = path.read_text()
        path.write_text(text.rsplit('\t', 1)[0] + '\t2\n')
        second = ep.read_neighborhoods(self.inputs, frame)
        self.assertEqual(second[0].center_index, 2)
        self.assertEqual([position for _, position, _ in ep.center_targets(second, frame, None)], [0, 2])
        for index in [-1, 1, 3]:
            path.write_text(text.rsplit('\t', 1)[0] + f'\t{index}\n')
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'order or center differs'):
                ep.read_neighborhoods(self.inputs, frame)
        path.write_text(text.replace('+A;-B;+A', '-B;+A;+A'))
        with self.assertRaisesRegex(ValueError, 'order or center differs'):
            ep.read_neighborhoods(self.inputs, frame)

    def test_indexed_hash_restores_same_query_name_rep_at_both_positions(self):
        frame = fixtures.neighborhood('a', 0).assign(nn_hashes='hash')
        rows = self.inputs_for(frame)
        mapping = {'hash': [['query', 'A', 'restored', 0], ['query', 'A', 'restored', 2]]}
        path = ep.run_embedding_batches(rows, frame, mapping, self.root / 'glm_outputs',
                                        self.vectors, 1, 5)
        output = pd.read_csv(path, sep='\t')
        self.assertEqual(output.nn_center_id.tolist(), [0, 2])
        self.assertEqual(output['0'].tolist(), [0, 2])
        self.assertEqual(output.neighborhood_name.tolist(), ['restored', 'restored'])
        self.assertEqual(len(output.drop_duplicates(['query', 'neighborhood_name', 'rep'])), 1)
        mapping['hash'][1][3] = 1
        with self.assertRaisesRegex(ValueError, 'does not match representative'):
            ep.center_targets(rows, frame, mapping)

    def test_partial_batches_resume_once_per_neighborhood_and_manifest_version(self):
        frame = pd.concat([duplicated_neighborhood('a'), fixtures.neighborhood('b', 1),
                           duplicated_neighborhood('c').assign(strand='-')])
        rows = self.inputs_for(frame)
        out = self.root / 'glm_outputs'
        calls = []
        def embed(batch):
            calls.append([r.neighborhood_name for r in batch])
            if len(calls) == 2:
                raise RuntimeError('interruption')
            return self.vectors(batch)
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            ep.run_embedding_batches(rows, frame, None, out, embed, 2, 5)
        self.assertEqual(calls, [['a', 'b'], ['c']])
        with patch.object(self, 'vectors', wraps=self.vectors) as remaining:
            result = ep.run_embedding_batches(rows, frame, None, out, remaining, 2, 5, resume=True)
            self.assertEqual(remaining.call_count, 1)
            self.assertEqual(len(remaining.call_args.args[0]), 1)
        output = pd.read_csv(result, sep='\t')
        self.assertEqual(output['0'].tolist(), [0, 2, 101, 200, 202])
        with patch.object(self, 'vectors', side_effect=AssertionError('recomputed')) as cached:
            ep.run_embedding_batches(rows, frame, None, out, cached, 2, 5, resume=True)
        manifest_path = out / 'embedding_batches.json'
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest['version'], 2)
        manifest.pop('version')
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'predate duplicate-center support'):
            ep.run_embedding_batches(rows, frame, None, out, self.vectors, 2, 5, resume=True)

    def test_prediction_and_meta_classifier_preserve_occurrence_metadata(self):
        frame = duplicated_neighborhood()
        rows = self.inputs_for(frame)
        embeddings = ep.center_dataframe(rows, self.vectors(rows), ep.center_targets(rows, frame, None))
        labels = SimpleNamespace(classes_=['x', 'y'])
        class Classifier(torch.nn.Module):
            def load_state_dict(self, state):
                pass
            def forward(self, values):
                return torch.stack([values[:, 0], -values[:, 0]], dim=1)
        with patch.object(nc.clf, 'FullyConnectedNN', return_value=Classifier()), \
             patch.object(torch, 'load', return_value={}), patch.object(nc, 'DEVICE', torch.device('cpu')):
            predictions = nc.get_embed_preds(embeddings, 'unused', labels, SimpleNamespace(glm_bs=1))
        pd.testing.assert_frame_equal(predictions[ep.METADATA], embeddings[ep.METADATA])
        self.assertNotEqual(predictions.x.iloc[0], predictions.x.iloc[1])
        structure = pd.DataFrame({'query': ['duplicated-query'], 'x': [.2], 'y': [.8]})
        joined = predictions.merge(structure, on='query', how='left')
        outer = self
        class MetaClassifier:
            def predict_proba(self, features):
                outer.assertEqual(features.columns.tolist(), ['0_x', '1_x', '0_y', '1_y'])
                np.testing.assert_allclose(features.iloc[:, :2], predictions[['x', 'y']])
                return features.iloc[:, :2].to_numpy()
        weights = self.root / 'weights'
        weights.write_bytes(b'unused')
        with patch.object(clf_models.pkl, 'load', return_value=MetaClassifier()):
            integrated = clf_models.meta_classifier(joined, weights, labels)
        pd.testing.assert_frame_equal(integrated[ep.METADATA], embeddings[ep.METADATA])


if __name__ == '__main__':
    unittest.main()
