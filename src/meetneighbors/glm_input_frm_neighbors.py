import pandas as pd
from Bio import SeqIO
import shutil
import numpy as np

from dataclasses import dataclass
import glob
import subprocess
from pathlib import Path
from meetneighbors.neighborhood_order import ordered_occurrences, query_labels

@dataclass(frozen=True)
class Neighborhood:
    source_file: str
    neighborhood_name: str
    queryname: tuple[str, ...]
    proteins: tuple[str, ...]
    center_index: int
    center_id: str


def link_genome_fasta(source, input_dir):
    """Link either combined FASTA variant, repairing old relative symlinks."""
    source = Path(source).resolve(strict=True)
    # Keep the historical input filename for compatibility with saved runs.
    destination = Path(input_dir) / 'combined_fastas_clust_rep.fasta'
    if destination.is_symlink():
        if destination.resolve() == source:
            return destination
        destination.unlink()
    destination.symlink_to(source)
    return destination


def get_glm_input(**kwargs):
    # This function organizes neighborhoods from mmseqs_clu_df into a tsv for gLM input
    # col1 should be the neighborhood name, col2 all proteins in neighborhood (along w/ strand info), col3 index of neighborhood protein center, other columns are of metadata (like VFcat, subcat, non_vf, etc..)

    query = kwargs.get("query",None)
    uniq_neighborhoods_d = kwargs.get("uniq_neighborhoods_d",None)
    neighborhood_res = kwargs.get("neighborhood_res",None)
    mmseqs_clust = kwargs.get("mmseqs_clust",None)
    glm_input_dir = kwargs.get("glm_input_dir",None)
    vfdb = kwargs.get("vfdb",None) # here for compatibility with chop_genome
    logger = kwargs.get("logger",None)
    args = kwargs.get("args",None)
    combinedfasta_ref = kwargs.get("combinedfasta", f"{args.out}combined_fastas_clust_rep.fasta") # need to be after args object retrieval

    pd.options.mode.chained_assignment = None

    if neighborhood_res is not None:
        if query not in list(neighborhood_res['query']): # skip if the vf was filtered out b/c of lack of hits decided by args.min_hits
            return []

    # only keep queries that made it through filtering in main.py to reduce computations   
    if uniq_neighborhoods_d: # here for compatiblity with chop-genome, since it doesn't currently make this object
        neighborhood_grp = uniq_neighborhoods_d[query]
    
    else:
        neighborhood_grp = set(mmseqs_clust[mmseqs_clust['query']==query]['neighborhood_name']) # i THINK this should be fine for chop-genome b/c the query col should be the same as the VF_center, so there shouldnt be neighborhoods w/ overlapping queries 

    mmseqs_clust = mmseqs_clust[mmseqs_clust['neighborhood_name'].isin(neighborhood_grp)]

    mmseqs_clust['strand_neighborhood'] = mmseqs_clust['strand'] + mmseqs_clust['rep'] # each protein in neighborhoods will be reprented by its cluster representative
    mmseqs_clust_grp = mmseqs_clust.groupby('neighborhood_name')

    nn_objs = []
    df_name = query
    
    neighborhood_name_prots = {}
    for neighborhood_name in sorted(neighborhood_grp):
        raw = mmseqs_clust_grp.get_group(neighborhood_name)
        ordered = ordered_occurrences(raw)
        center = ordered.VF_center.iloc[0] + '!!!' + ordered.gff.iloc[0]
        indices = np.flatnonzero(ordered['prot_gffname'].eq(center))
        if len(indices) == 0:
            raise ValueError(f"{neighborhood_name}: no VF center found")
        if len(indices) > 1:
            logger.info(f"{neighborhood_name}: multiple VF centers found, likely a duplication event")
        index = int(indices[0])
        labels = query_labels(raw.loc[raw.prot_gffname.eq(center), 'query'])
        proteins = tuple(ordered['strand'] + ordered['rep'])
        nn_objs.append(Neighborhood(
            f"{args.out}{glm_input_dir}/{df_name}.tsv", neighborhood_name,
            tuple(labels), proteins, index, ordered.rep.iloc[index]))
        neighborhood_name_prots[neighborhood_name] = [';'.join(proteins), index]

    df = pd.DataFrame.from_dict(neighborhood_name_prots).T
    df.columns=['neighborhood','VF_center_index']

    df.to_csv(f"{args.out}{glm_input_dir}/{df_name}.tsv",sep='\t',header=False,mode="x")    
    
    return nn_objs

def get_glm_fasta_input(fasta_path,args,glm_input_dir,**kwargs):
    query_grp = kwargs.get("query_grp",None)
    protids = kwargs.get('protids',None)
    chunk_it = kwargs.get('it',None)
    
    if protids: 
        og_fasta = SeqIO.parse(fasta_path,"fasta")
        fasta = filter(lambda x: x.id.split('|')[-1] in list(protids),og_fasta)
        fasta_outpath = f"{glm_input_dir}combined_chunk_{str(chunk_it)}.fasta"

    elif query_grp:
        df_name,grp = query_grp[0],query_grp[1]
        fasta = []
        valid_reps = set(grp.rep.values)
        for rec in SeqIO.parse(fasta_path,"fasta"):
            rec.id = rec.id.split('|')[-1]
            if rec.id in valid_reps:
                fasta.append(rec)
        fasta_outpath = f"{glm_input_dir}{df_name}.fasta"
        
    with open(fasta_outpath,"x") as handle: #tsv and fasta files should have the same name
        SeqIO.write(fasta, handle, "fasta")
    return

def concat_tsv_fastas(directory, glm_input2_out, chunk, i):
    # this is a tempory fix for slow embedding computations...I discovered that embeddings are slow b/c its starting then processing one file at a time
    # the real fix should modify glm_input_frm_neighbors.py so that this function function doesn't need to happen
    tsv_files = glob.glob(f"{directory}/*.tsv")
    # fasta_files = glob.glob(f"{directory}/*.fasta")
    chunk_files = [f for f in tsv_files if f.split('/')[-1].split('.tsv')[0] in chunk]

    output_file = f"{glm_input2_out}combined_chunk_{str(i)}.tsv"

    # Use cat to concatenate files directly
    cmd = ['cat'] + chunk_files
        
    with open(output_file, 'w') as outfile:
        subprocess.run(cmd, stdout=outfile, check=True)
        
    print(f"Processed chunk {i}: {len(chunk_files)} files")
    return 
