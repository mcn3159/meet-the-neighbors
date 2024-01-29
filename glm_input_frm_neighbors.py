from collections import defaultdict 
import pandas as pd
from Bio import SeqIO
import subprocess

def get_glm_input(query,uniq_neighborhoods_d,out_dir):
    #lets say the input is a dask bag where the contents are dictionary records with key being query, and value being neighborhoods_l 
    # one fasta per query? Should keep df mapping query to its descriptions
    # tsv where each contig is a neighborhood name as col 1, and the col2 is the fasta entry of each neighborhood in the contig, should be sorted by position of protein on chromosome
    # the tsv mentioned above should also have similar name to fasta
    subprocess.run(f"cat {out_dir}combined_fasta_partition* > {out_dir}all_neighborhoods.fasta",shell=True,check=True)
    subprocess.run(f"rm {out_dir}combined_fasta_partition*",shell=True,check=True)
    fasta = SeqIO.parse(f"{out_dir}all_neighborhoods.fasta","fasta")
    neighborhood_grp = uniq_neighborhoods_d[query]
    fasta_filt = [(rec,int(rec.description.split("--start")[1].split('--strand')[0])) for rec in fasta if rec.id.split('----',1)[1] in neighborhood_grp] # [(seq_record,position),...]
    fasta_filt = sorted(fasta_filt,key = lambda x:x[1])
    neighborhood_name_prots = defaultdict(str)
    for rec in fasta_filt: # each neighborhood for each query sorted by name of neighborhood
        neighborhood_name = rec[0].id.split('----',1)[1]
        strand = rec[0].description.split("--strand")[1]
        if len(neighborhood_name_prots[neighborhood_name]) != 0:
            rec.id = ',' + rec[0].id
        neighborhood_name_prots[neighborhood_name] = neighborhood_name_prots[neighborhood_name] + strand + rec.id

    return pd.DataFrame.from_dict(neighborhood_name_prots,orient='index').to_csv(f"{out_dir}{query}_{neighborhood_name}.tsv",sep='\t',index=False,header=False)