import subprocess
import os
import pandas as pd
import glob
# script to call mmseqs functions

def mmseqs_createdb(args):
    subprocess.run(["mmseqs","createdb",args.query_fasta,f"{args.out}queryDB","-v","2"],check=True)
    if not args.genomes_db:
        subprocess.run(f"mmseqs createdb {args.genomes}*.fa* {args.out}genomesdb -v 2",shell=True,check=True) # assuming protein fastas end with with .faa or .fasta
    return 

def mmseqs_search(args,genomes_db):
    gpu = -1 # here until mmseqs team fixes the tset issue with gpu accelerated search
    if gpu > 0:
        subprocess.run(f"mmseqs makepaddedseqdb {genomes_db} {args.out}genomesdb_gpu --threads {args.threads}",shell=True,check=True)
        subprocess.run(f"rm {genomes_db}.*",shell=True,check=True) # base db file made from mmseqs is kept b/c if i rm it with the same strategy, I remove the gpu db as well
        subprocess.run(f"mmseqs search {args.out}queryDB {args.out}genomesdb_gpu {args.out}vfs_in_genomes {args.out}tmp_search --min-seq-id {args.seq_id} --cov-mode 0 -c {args.cov} -v 2  --split-memory-limit {int(args.mem * (2/3))}G --alignment-mode 3 --gpu {args.gpu}",shell=True,check=True)
    else:
        subprocess.run(f"mmseqs search {args.out}queryDB {genomes_db} {args.out}vfs_in_genomes {args.out}tmp_search --min-seq-id {args.seq_id} --cov-mode 0 -c {args.cov} -v 2 --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads} --alignment-mode 3 --start-sens 1 --sens-steps 3 -s 7",
                       shell=True,check=True)

    subprocess.run(["mmseqs", "convertalis", f"{args.out}queryDB", genomes_db, f"{args.out}vfs_in_genomes", f"{args.out}vfs_in_genomes.tsv", "--format-output", "query,target,evalue,pident,qcov,fident,alnlen,qheader,theader,tset,tsetid"] 
        ,check=True)
    return

def mmseqs_cluster(args,glm_inputs=True):
        if (args.resume and not os.path.isfile(f"{args.out}combined_fastas_clust_res.tsv")) or (not args.resume): # don't recreate clus_res.tsv if it already exists
            if isinstance(args.genome_tsv,pd.DataFrame):
                subprocess.run(f"mmseqs createdb {' '.join(list(args.genome_tsv['protein']))} {args.out}combined_fastas_db -v 2",shell=True,check=True) # should probably put the createdb call in the mmseqs_createdb() function
            elif args.genomes and not (args.query_fasta or args.prot_genome_pairs):
                genome_faas = glob.glob(args.genomes+'*.faa')
                genome_gffs = glob.glob(args.genomes+'*.gff')
                try:
                    assert len(genome_faas) == len(genome_gffs), "Number of .faa and .gff files do not match. Trying with .fastas"
                    protein_file_ending = ".faa"
                except AssertionError:
                    genomes_fastas = glob.glob(args.genomes+'*.fasta')
                    assert len(genomes_fastas) == len(genome_gffs), "Number of .fasta and .gff files do not match. When querying with --genomes make sure all protein files have a .faa ending, or all have a .fasta ending. And the number of protein files match the number of .gff files"
                    protein_file_ending = ".fasta"
                subprocess.run(f"mmseqs createdb {args.genomes}*{protein_file_ending} {args.out}combined_fastas_db -v 2",shell=True,check=True) # should probably put the createdb call in the mmseqs_createdb() function

            else:
                subprocess.run(f"mmseqs createdb {args.out}combined_fasta_partition* {args.out}combined_fastas_db -v 2",shell=True,check=True)
            # hard coded some clustering params b/c the goal is to reduce redundant proteins
            subprocess.run(f"mmseqs cluster {args.out}combined_fastas_db {args.out}combined_fastas_clust --cov-mode 0 -c 0.90 --min-seq-id 0.90 --similarity-type 2 -v 2 --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads} {args.out}tmp_clust",
                            shell=True,check=True)
            subprocess.run(f"mmseqs createtsv {args.out}combined_fastas_db {args.out}combined_fastas_db {args.out}combined_fastas_clust {args.out}combined_fastas_clust_res.tsv",shell=True,check=True)

        if glm_inputs:
            if not os.path.isfile(f"{args.out}combined_fastas_clust_rep.fasta"): # save time if resuming
                subprocess.run(f"mmseqs createsubdb {args.out}combined_fastas_clust {args.out}combined_fastas_db {args.out}combined_fastas_clust_rep",shell=True,check=True)
                subprocess.run(f"mmseqs convert2fasta {args.out}combined_fastas_clust_rep {args.out}combined_fastas_clust_rep.fasta",shell=True,check=True)
        return