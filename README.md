# meet-the-neighbors
Medium scale identification of genomic neighbors in bacterial genomes. This program extracts neighborhoods from a set gffs and its respective protein fasta file with the same name. Extracted neighborhoods will be outputted in a fasta file, with the protein identifier denoting the neighborhood name and protein ID. Multiple neighborhoods can be placed into the same fasta file. 

To determine which protein entry in the fasta file belongs to what neighborhood, each entry in the fasta file haas a header with the following naming scheme:

"Protein_id"!!!"Protein_center"!!!"gff_with_neighborhood"!!!"Contig_with_neighborhoods"!!!"Neighborhood_start_position"!!!"Neighborhood_end_position"!!!"Protein_start_position"!!!"strand"

### Main functionalities

- Extracts neighborhoods into a protein fasta
- Clusters similar neighborhoods found with the same intial query using DBscan (epsilon=0.15,min_samples=3)
- Plots Neighborhood results in a bubble chart with size of bubbles being 100/(# number of neighborhoods outside of cluster+1)
- Outputs neighborhoods in compatible format for a genomic language model (https://github.com/y-hwang/gLM)
- To be continued...

### Running Program
```
git clone
```

Activate the conda environment
```
conda env create -f environment.yml
conda activate meet-the-neighbors
```

Basic run command 
```
python main.py extract_neighbors --query_fasta dir/to/queries.fasta --genomes dir/to/genomes_to_search --out dir/to/out_directory --plot
```
Each genome to extract the neighborhoods from must occur in a gff and protein fasta formats, with the same file name followed by an underscore _genomic or _protein respectively.
For example:

```
cd genomes/

GCA_000307975.2_ASM30797v2_genomic.gff
GCA_000307975.2_ASM30797v2_protein.faa
...
```

### Context-only GLM embeddings

Use the inputs prepared by the `context_only.ipynb` loop:

```bash
meetneighbors context_only \
  --input-dir /path/to/masked_embeds \
  --out /path/to/context_only_results \
  --glm_bs 100 \
  --device auto
```

The input directory must contain `glm_inputs/*.tsv` and
`glm_outputs/all_glm_input_prots_reps.esm.embs.pkl`. Each headerless TSV row is
`neighborhood_name<TAB>semicolon-separated stranded protein IDs<TAB>zero-based center index`.
Protein IDs start with `+` or `-`. The default `--input-mode masked` expects the
notebook's terminal `_masked` suffix on each center ID. To use unmodified
meet-the-neighbors outputs directly, pass `--input-mode original` and point
`--input-dir` at the original run directory. No ESM recomputation is needed.

For each neighborhood, the command replaces the center's **raw 1,280-dimensional
ESM vector with -1 before normalization**. Normalization therefore changes those
values; GLM does not receive literal -1 features. The center's strand and position
are retained, and neighboring occurrences of the same protein keep their own ESM
vectors. Centers are identified by the third TSV column, not by a substring search.

The command uses the existing GLM weights and normalization factors. Set
`MEETNEIGHBORS_GLM_BIN=/path/to/glm.bin` if the checkpoint is not in the repository's
`src/meetneighbors/predictvfs/glm/model/` directory. `--device auto` selects CUDA
when available, otherwise CPU; `--device cpu` and `--device cuda` explicitly choose
the device. CUDA uses mixed precision. CPU uses float32 inference. Input staging
uses the existing batcher's float16 precision. Neighborhoods must contain 1–30 proteins.

The output is `context_only_layer9.tsv`, with `source_file`, one-based `source_row`,
`neighborhood_name`, `rep` (original center ID), and embedding columns `0`–`1279`.
Only center embeddings are saved, in sorted filename order and original row order.
Repeated proteins in different neighborhoods remain separate rows. Layer 9 means
hidden-state index 9, where index 0 is the embedding-stage output. No classifier
or neighborhood-hash expansion is run.

The source files remain unchanged. Existing output tables are rejected. Invalid
center indices, strand prefixes, missing neighbor embeddings, invalid ESM vectors,
and oversized neighborhoods are reported before model inference. Paths do not
require trailing slashes. Original center IDs are restored by removing just one
terminal `_masked` suffix in masked mode. The unused notebook `allmaskedembeds`
branch is not part of this implementation.
