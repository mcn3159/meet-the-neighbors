"""Genome-only FASTA layouts and duplicate centers, without loading large models."""
import logging
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Bio import SeqIO
import dask
import numpy as np
import pandas as pd
import torch

from meetneighbors import embedding_pipeline as ep
from meetneighbors import glm_input_frm_neighbors as gi
from meetneighbors import plot_map_neighborhood_res as pn
from meetneighbors import ring_mmseqs as mm
from meetneighbors.main import get_parser, workflow
from meetneighbors.predictvfs import neighborhood_classification as nc
from meetneighbors.predictvfs.glm import glm_embed as ge
from test_duplicate_centers import duplicated_neighborhood

TEST_DATA = Path(__file__).resolve().parent / 'test_data'

class GenomeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        # Match the reported ../output layout, without enough parent traversals
        # to reach / (which would accidentally make the old link resolve).
        work = self.root / 'work'
        work.mkdir()
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(work)
        self.logger = logging.getLogger('genome-input-tests')
        torch.set_num_threads(1)

    def test_link_both_fasta_names_relative_output_and_existing_broken_link(self):
        for filename in ('combined.fasta', 'combined_fastas_clust_rep.fasta'):
            with self.subTest(filename=filename):
                out = self.root / filename.replace('.', '-')
                inputs = out / 'glm_inputs'
                inputs.mkdir(parents=True)
                source = out / filename
                source.write_text('>A\nMALWMRLLPLL\n')
                relative_source = os.path.relpath(source)
                destination = inputs / 'combined_fastas_clust_rep.fasta'
                # Reproduce exactly what the previous os.symlink call wrote.
                destination.symlink_to(relative_source)
                self.assertFalse(destination.exists())
                gi.link_genome_fasta(relative_source, os.path.relpath(inputs))
                self.assertEqual(destination.read_bytes(), source.read_bytes())
                self.assertEqual(ep.file_digest(destination), ep.file_digest(source))
                gi.link_genome_fasta(relative_source, inputs)  # idempotent
                self.assertEqual(destination.resolve(), source.resolve())

    def test_hash_keeps_duplicate_centers_and_collapses_redundant_search_rows(self):
        frame = duplicated_neighborhood()
        frame = pd.concat([frame, frame.iloc[[0]].assign(query=[['first', 'second']]),
                           frame.iloc[[2]], duplicated_neighborhood('b')], ignore_index=True)
        frame = frame.sample(frac=1, random_state=10)
        clusters, mapping = pn.hash_neighborhoods(frame, SimpleNamespace())
        self.assertEqual(set(mapping), {hash(('A', 'B', 'A'))})
        expected = {('duplicated-query', 'A', name, position)
                    for name in ('a', 'b') for position in (0, 2)}
        expected.update({('first', 'A', 'a', 0), ('second', 'A', 'a', 0)})
        self.assertEqual({tuple(entry) for entries in mapping.values() for entry in entries}, expected)
        self.assertEqual(dict(zip(clusters.start, clusters.nn_idx)), {2: 0, 10: 1, 100: 2})
        inputs = self.root / 'glm_inputs'
        inputs.mkdir()
        name = clusters.neighborhood_name.iloc[0]
        gi.get_glm_input(query=name, uniq_neighborhoods_d={name: {name}}, mmseqs_clust=clusters,
                         glm_input_dir='glm_inputs', logger=self.logger,
                         args=SimpleNamespace(out=str(self.root) + '/'))
        rows = ep.read_neighborhoods(inputs, clusters)
        targets = ep.center_targets(rows, clusters, mapping)
        vectors = np.broadcast_to(np.arange(3)[None, :, None], (1, 3, 1280))
        output = ep.center_dataframe(rows, vectors, targets)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(output), len(expected))
        np.testing.assert_array_equal(output[0], output.nn_center_id)

    def test_genomes_cli_relative_output_high_memory_layout_clustered_and_unclustered(self):
        genomes = self.root / 'genomes'
        genomes.mkdir()
        (genomes / 'g.faa').write_text('>A\nMALWMRLLPLL\n>B\nMKKLALALALA\n>C\nMTTGGVVAAAL\n')
        (genomes / 'g.gff').write_text('##gff-version 3\n' + ''.join(
            f'ctg\ttest\tCDS\t{start}\t{start + 65}\t.\t+\t0\tID={protein};protein_id={protein};locus_tag={protein}\n'
            for protein, start in [('A', 2), ('B', 100), ('A', 200), ('C', 300)]))
        checkpoint = self.root / 'checkpoint'
        checkpoint.write_text('test')
        norms = {'norm.pkl': {'mean': np.zeros(1280), 'std': np.ones(1280)},
                 'labelbinarizer_vfcategories.obj': SimpleNamespace(classes_=['x', 'y'])}

        def cluster(args):
            # Only external MMseqs is stubbed; parse/merge its real file layout.
            out = Path(args.out)
            shutil.copyfile(genomes / 'g.faa', out / 'combined_fastas_clust_rep.fasta')
            (out / 'combined_fastas_clust_res.tsv').write_text('A\tA\nB\tB\nC\tC\n')

        def plm(**kwargs):
            records = list(kwargs['seq_records'])
            self.assertEqual({record.id for record in records}, {'A', 'B', 'C'})
            return [(record.id, torch.zeros(1280)) for record in records]

        def embed(reps, model, rows, mean, std, maximum, layer):
            self.assertEqual(layer, 3)
            nc.prepare_batch(rows, reps, mean, std, maximum)
            return np.broadcast_to(np.arange(maximum)[None, :, None], (len(rows), maximum, 1280)).copy()

        for clustered in (False, True):
            for hashed in (False, True):
                with self.subTest(clustered=clustered, hashed=hashed):
                    out = self.root / f'out-{clustered}-{hashed}'
                    relative_out = os.path.relpath(out)
                    argv = ['meetneighbors', 'extract_neighbors', '--genomes', str(genomes),
                            '-o', relative_out, '--threads', '1', '--mem', '100', '--glm_bs', '2',
                            '-ns', '30000', '-ig', '10000', '--max_prots', '5', '--head_on']
                    if clustered:
                        argv += ['--cluster']
                    if hashed:
                        argv += ['--memory_optimize']
                    with patch('sys.argv', argv), dask.config.set(scheduler='synchronous'), \
                         patch('meetneighbors.main.get_logger', return_value=self.logger), \
                         patch.object(mm, 'mmseqs_cluster', side_effect=cluster), \
                         patch.object(nc.plm, 'get_distributed_backend_and_device', return_value=('gloo', 'cpu', False)), \
                         patch.object(nc.plm, 'load_plm', return_value=(object(), object())), \
                         patch.object(nc.plm, 'run_plm', side_effect=plm) as run_plm, \
                         patch.object(ge, 'load_model', return_value=object()) as load, \
                         patch.object(nc, 'embed_sequences', side_effect=embed) as embeddings, \
                         patch('meetneighbors.predictvfs.loader.load_pickle', return_value=norms), \
                         patch('meetneighbors.predictvfs.loader.get_glm_model_path', return_value=checkpoint):
                        workflow(get_parser())
                        load.assert_called_once()
                        run_plm.assert_called_once()
                    self.assertEqual([len(call.args[2]) for call in embeddings.call_args_list],
                                     [1] if hashed else [2, 1])
                    frame = pd.read_csv(out / 'glm_embeds.tsv', sep='\t')
                    self.assertEqual(set(zip(frame['query'], frame.nn_center_id)),
                                     {('A', 0), ('A', 2), ('B', 1), ('C', 3)})
                    self.assertEqual(len(frame), 4)
                    self.assertEqual(set(frame.glminput_tsv), {'g'})
                    np.testing.assert_array_equal(frame['0'], frame.nn_center_id)
                    self.assertEqual(list(frame.columns), ep.METADATA + [str(i) for i in ep.DIMENSIONS])
                    filename = 'combined_fastas_clust_rep.fasta' if clustered else 'combined.fasta'
                    link = out / 'glm_inputs' / 'combined_fastas_clust_rep.fasta'
                    self.assertEqual(link.resolve(), (out / filename).resolve())
                    self.assertEqual({r.id for r in SeqIO.parse(link, 'fasta')}, {'A', 'B', 'C'})
                    # Repair an old run's broken link before fingerprinting, then
                    # reuse the completed vectors in both downstream commands.
                    link.unlink()
                    link.symlink_to(str(Path(relative_out) / filename))
                    self.assertFalse(link.exists())
                    classifier = torch.nn.Linear(1280, 2)
                    for command in ('extract_neighbors', 'predictvf'):
                        with patch('sys.argv', [argv[0], command, *argv[2:], '--resume']), \
                             patch('meetneighbors.main.get_logger', return_value=self.logger), \
                             patch.object(nc, 'get_plm_embeds', side_effect=AssertionError('resume recomputed')), \
                             patch('meetneighbors.predictvfs.loader.load_pickle', return_value=norms), \
                             patch('meetneighbors.predictvfs.loader.get_glm_model_path', return_value=checkpoint), \
                             patch('meetneighbors.predictvfs.loader.load_clf_models', return_value={'nn_clf': 'unused'}), \
                             patch.object(nc.clf, 'FullyConnectedNN', return_value=classifier), \
                             patch.object(torch, 'load', return_value=classifier.state_dict()), \
                             patch.object(nc, 'DEVICE', torch.device('cpu')):
                            workflow(get_parser())
                    predictions = pd.read_csv(out / 'neighborhood_based_predictions.tsv', sep='\t')
                    pd.testing.assert_frame_equal(predictions[ep.METADATA], frame[ep.METADATA])
                    from pipeline_checks import check_context_output
                    check_context_output(self, out)


@unittest.skipUnless(os.environ.get('MEETNEIGHBORS_TEST_REAL_GENOME') == '1',
                     'opt-in end-to-end genome test with local ESM/GLM/classifier checkpoints')
class RealGenomeTests(unittest.TestCase):
    def test_nr2335_extraction_prediction_and_resume_with_real_models(self):
        """Run actual extraction, ESM, layer-3 GLM, classifier and MMseqs on CPU."""
        from meetneighbors.predictvfs import loader
        from meetneighbors import compute_umap
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            genomes = root / 'genomes'
            genomes.mkdir()
            # Keep original CDS records/coordinates, including both occurrences
            # of NR2335_000081, and real protein sequences for three genes.
            selected = []
            with (TEST_DATA / 'genomes' / 'NR2335.gff').open() as handle:
                for line in handle:
                    fields = line.rstrip().split('\t')
                    if (len(fields) == 9 and fields[0] == 'cluster_001' and fields[2] == 'CDS'
                            and 84654 <= int(fields[3]) <= 86898):
                        selected.append(line)
            self.assertEqual(len(selected), 4)
            gff = genomes / 'NR2335.gff'
            gff.write_text('##gff-version 3\n' + ''.join(selected))
            ids = {'NR2335_000081', 'NR2335_000082', 'NR2335_000083'}
            records = [r for r in SeqIO.parse(TEST_DATA / 'genomes' / 'NR2335.faa', 'fasta')
                       if r.id.split('|')[-1] in ids]
            self.assertEqual(len(records), 3)
            fasta = genomes / 'NR2335.fasta'
            SeqIO.write(records, fasta, 'fasta')
            genome_tsv = root / 'genomes.tsv'
            genome_tsv.write_text(f'NR2335\t{fasta}\t{gff}\n')
            labels = loader.load_pickle()['labelbinarizer_vfcategories.obj'].classes_.tolist()
            original_cwd = Path.cwd()
            work = root / 'work'
            work.mkdir()
            os.chdir(work)
            try:
                for mode in ('genomes', 'genome_tsv', 'query_fasta'):
                    clustered = mode == 'genome_tsv'
                    with self.subTest(mode=mode):
                        out = root / f'output-{mode}'
                        command = 'predictvf' if clustered else 'extract_neighbors'
                        argv = ['meetneighbors', command, '-o', f'../{out.name}', '--threads', '1',
                                '--mem', '8' if clustered else '100', '--glm_bs', '2',
                                '-ns', '30000', '-ig', '10000']
                        if clustered:
                            argv += ['--genome_tsv', str(genome_tsv), '--cluster', '--memory_optimize']
                        else:
                            argv += ['--genomes', str(genomes)]
                            if mode == 'query_fasta':
                                argv += ['--query_fasta', str(fasta), '--memory_optimize']
                        # Only execution settings are changed: all models, their
                        # weights, FASTA processing, clustering and predictions are real.
                        with patch('sys.argv', argv), dask.config.set(scheduler='synchronous'), \
                             patch.object(nc, 'DEVICE', torch.device('cpu')):
                            workflow(get_parser())
                        embeddings = pd.read_csv(out / 'glm_embeds.tsv', sep='\t')
                        expected = {('NR2335_000081', 0), ('NR2335_000081', 1),
                                    ('NR2335_000082', 2), ('NR2335_000083', 3)}
                        self.assertEqual(set(zip(embeddings['query'], embeddings.nn_center_id)), expected)
                        self.assertEqual(len(embeddings), 4)
                        values = embeddings[ep.embedding_columns(embeddings)].to_numpy()
                        self.assertTrue(np.isfinite(values).all())
                        copies = embeddings.loc[embeddings['query'].eq('NR2335_000081')].sort_values('nn_center_id')
                        self.assertFalse(np.array_equal(copies[ep.embedding_columns(copies)].iloc[0],
                                                       copies[ep.embedding_columns(copies)].iloc[1]))
                        clusters = pd.read_csv(out / 'clust_res_in_neighborhoods' / 'mmseqs_clust.tsv', sep='\t')
                        rows = ep.read_neighborhoods(out / 'glm_inputs', clusters)
                        mapping = None
                        if clustered or mode == 'query_fasta':
                            with (out / 'nnhash_mapping.obj').open('rb') as handle:
                                mapping = pickle.load(handle)
                        # Independently recompute one context at a time to verify
                        # each saved row selected the exact model output position.
                        with (out / 'glm_outputs' / 'all_glm_input_prots_reps.esm.embs.pkl').open('rb') as handle:
                            reps = {name: [vector, name] for name, vector in pickle.load(handle)}
                        norm = loader.load_pickle()['norm.pkl']
                        model = ge.load_model(torch.device('cpu'))
                        with patch.object(nc, 'DEVICE', torch.device('cpu')):
                            for i, row in enumerate(rows):
                                vectors = nc.embed_sequences(reps, model, [row], norm['mean'], norm['std'], 30, layer=3)
                                for source, position, metadata in ep.center_targets(rows, clusters, mapping):
                                    if source == i:
                                        match = (embeddings[ep.METADATA] == pd.Series(metadata, index=ep.METADATA)).all(axis=1)
                                        self.assertEqual(int(match.sum()), 1)
                                        np.testing.assert_allclose(values[match.to_numpy()][0], vectors[0, position], rtol=1e-4, atol=1e-5)
                        del model, reps, vectors
                        before = {path.name: ep.file_digest(path) for path in ep.saved_batch_paths(out / 'glm_outputs')}
                        # Resume both public entry points; output files must stay identical.
                        for resumed_command in ('extract_neighbors', 'predictvf'):
                            with patch('sys.argv', [argv[0], resumed_command, *argv[2:], '--resume']), \
                                 dask.config.set(scheduler='synchronous'), \
                                 patch.object(nc, 'DEVICE', torch.device('cpu')), \
                                 patch.object(nc, 'get_plm_embeds', side_effect=AssertionError('resume recomputed ESM')), \
                                 patch.object(ge, 'load_model', side_effect=AssertionError('resume recomputed GLM')):
                                workflow(get_parser())
                        self.assertEqual(before, {path.name: ep.file_digest(path) for path in ep.saved_batch_paths(out / 'glm_outputs')})
                        predictions = pd.read_csv(out / 'neighborhood_based_predictions.tsv', sep='\t')
                        pd.testing.assert_frame_equal(predictions[ep.METADATA], embeddings[ep.METADATA])
                        self.assertTrue(np.isfinite(predictions[labels]).all().all())
                        np.testing.assert_allclose(predictions[labels].sum(axis=1), 1, atol=1e-6)
                        unpacked = compute_umap.get_glm_embeddf(compute_umap.unpack_embeddings(
                            str(out / 'glm_outputs'), str(out / 'glm_inputs'), clusters))
                        pd.testing.assert_frame_equal(unpacked[ep.METADATA].reset_index(drop=True), embeddings[ep.METADATA])
                        from pipeline_checks import check_context_output
                        check_context_output(self, out, real=True)
                        print(f'REAL_PIPELINE: mode={mode}, 4 center rows, real ESM/GLM/classifier/context-only, resume verified', flush=True)
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
