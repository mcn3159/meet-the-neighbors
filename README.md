# meet the neighbors
Medium scale identification of genomic contexts and virulence factors in bacterial genomes. This program extracts neighborhoods from a set gffs and its respective protein fasta file with the same name. Extracted neighborhoods will be outputted in a fasta file.

Schema for neighborhood ids:


`protein_id!!!seed_proteinid!!!gff!!!contig!!!neighborhood_start-neighborhood_end`

### Main functionalities

- Extracts neighborhoods of a query fasta from a given set of genomes
- Outputs neighborhoods in compatible format for a genomic language model (https://github.com/y-hwang/gLM)
- Predicts VFDB categories of whole genome or given fasta

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
ls genomes/

GCA_000307975.2_ASM30797v2_genomic.gff
GCA_000307975.2_ASM30797v2_protein.faa
...

