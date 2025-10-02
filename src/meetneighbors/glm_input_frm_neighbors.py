import pandas as pd
from Bio import SeqIO
import shutil
import numpy as np

from meetneighbors.predictvfs import neighborhood_classification as nc


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

    if neighborhood_res:
        if query not in list(neighborhood_res['query']): # skip if the vf was filtered out b/c of lack of hits decided by args.min_hits
            return

    # only keep queries that made it through filtering in main.py to reduce computations   
    if uniq_neighborhoods_d: # here for compatiblity with chop-genome, since it doesn't currently make this object
        neighborhood_grp = uniq_neighborhoods_d[query]
    
    else:
        neighborhood_grp = set(mmseqs_clust[mmseqs_clust['query']==query]['neighborhood_name']) # i THINK this should be fine for chop-genome b/c the query col should be the same as the VF_center, so there shouldnt be neighborhoods w/ overlapping queries 

    mmseqs_clust = mmseqs_clust[mmseqs_clust['neighborhood_name'].isin(neighborhood_grp)]

    mmseqs_clust['strand_neighborhood'] = mmseqs_clust['strand'] + mmseqs_clust['rep'] # each protein in neighborhoods will be reprented by its cluster representative
    mmseqs_clust_grp = mmseqs_clust.groupby('neighborhood_name')
    
    neighborhood_name_prots = {}
    for neighborhood_name in neighborhood_grp: # loop through neighborhoods to for gLM inputs
        mmseqs_clust_sub = mmseqs_clust_grp.get_group(neighborhood_name)
        # print("Size of mmseq_clust after filter by neighborhood_grp",len(mmseqs_clust_sub))
        mmseqs_clust_sub.drop_duplicates(subset=['start'],inplace=True) # in case something duplicate rows were made w/ earlier mmseqs merge
        mmseqs_clust_sub.reset_index(drop=True,inplace=True)
        center = mmseqs_clust_sub.VF_center.iloc[0] + '!!!' + mmseqs_clust_sub.gff.iloc[0] 
        mmseqs_clust_sub.sort_values(by='start',inplace=True) # this needs to be last in the loop. B/c reset_index reshuffles row locations
        ind = np.flatnonzero([mmseqs_clust_sub['prot_gffname']==center]) # index of the VF center in the neighborhood used for identifying its embedding in gLM output
        if len(ind) > 1:
            logger.warning(f"Multiple VF centers with the same name found in neighborhood: {neighborhood_name} ... grabing first vf_center")

        # assert int(mmseqs_clust_sub.start.iloc[-1]) -  int(mmseqs_clust_sub.start.iloc[0]) < args.neighborhood_size + 30000 # want to make sure that everything is working right, so if neighborhoods are much larger than expected, someting's wrong
        try:
            neighborhood_name_prots[neighborhood_name] = [';'.join(list(mmseqs_clust_sub.strand_neighborhood)),ind[0]]
        except TypeError:
            print("===== ",neighborhood_name," =====")
            print(mmseqs_clust_sub['prot_gffname'])
            print(mmseqs_clust_sub)
            raise Exception

    df = pd.DataFrame.from_dict(neighborhood_name_prots).T
    df.columns=['neighborhood','VF_center_index']

    # if vfdb:
    #     # want vf categories in the results so that downstream classification is easier
    #     neighborhood_res = neighborhood_res[neighborhood_res["query"] == query][['vf_name','vf_id','vf_subcategory','vf_category','vfdb_species','vfdb_genus']]
    #     df["vf_name"],df["vf_id"],df["vf_subcategory"],df["vf_category"],df["vfdb_species"],df["vfdb_genus"] = neighborhood_res.iloc[0][0],neighborhood_res.iloc[0][1],neighborhood_res.iloc[0][2],neighborhood_res.iloc[0][3],neighborhood_res.iloc[0][4],neighborhood_res.iloc[0][5]
    # else:
    #     df["vf_name"],df["vf_id"],df["vf_subcategory"],df["vf_category"],df["vfdb_species"],df["vfdb_genus"] = "non_vf","non_vf","non_vf","non_vf","non_vf","non_vf"
    #df_name = f"{neighborhood_res.iloc[0][1].replace(' ','_').replace(')','').replace('(','').replace('/','|')}_{str(random.randint(1,10000))}_{neighborhood_name}"
    df_name = query
    df.to_csv(f"{args.out}{glm_input_dir}/{df_name}.tsv",sep='\t',header=False,mode="x")    
    
    return 

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
        for rec in fasta_path:
            rec.id = rec.id.split('|')[-1]
            if rec.id in valid_reps:
                fasta.append(rec)
        fasta_outpath = f"{args.out}{glm_input_dir}/{df_name}.fasta"
        
    with open(fasta_outpath,"x") as handle: #tsv and fasta files should have the same name
        SeqIO.write(fasta, handle, "fasta")
    return

def concat_tsv_fastas(directory, glm_input2_out, chunk, i):
    # this is a tempory fix for slow embedding computations...I discovered that embeddings are slow b/c its starting then processing one file at a time
    # the real fix should modify glm_input_frm_neighbors.py so that this function function doesn't need to happen
    tsv_files = nc.glob.glob(f"{directory}/*.tsv")
    # fasta_files = nc.glob.glob(f"{directory}/*.fasta")
    chunk_files = [f for f in tsv_files if f.split('/')[-1].split('.tsv')[0] in chunk]

    output_file = f"{glm_input2_out}combined_chunk_{str(i)}.tsv"

    # Use cat to concatenate files directly
    cmd = ['cat'] + chunk_files
        
    with open(output_file, 'w') as outfile:
        nc.subprocess.run(cmd, stdout=outfile, check=True)
        
    print(f"Processed chunk {i}: {len(chunk_files)} files")
    return 