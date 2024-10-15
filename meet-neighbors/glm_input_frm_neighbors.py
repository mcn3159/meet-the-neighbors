import pandas as pd
from Bio import SeqIO
import random 
import numpy as np

def get_glm_input(query,uniq_neighborhoods_d,neighborhood_res,mmseqs_clust,glm_input_dir,logger,args):
    # This function organizes neighborhoods from mmseqs_clu_df into a tsv for gLM input
    # col1 should be the neighborhood name, col2 all proteins in neighborhood (along w/ strand info), col3 index of neighborhood protein center, other columns are of metadata (like VFcat, subcat, non_vf, etc..)

    pd.options.mode.chained_assignment = None
    if query not in list(neighborhood_res['query']): # skip if the vf was filtered out b/c of lack of hits decided by args.min_hits
        return

    # only keep queries that made it through filtering in main.py to reduce computations    
    neighborhood_grp = uniq_neighborhoods_d[query]
    mmseqs_clust = mmseqs_clust[mmseqs_clust['neighborhood_name'].isin(neighborhood_grp)]

    mmseqs_clust['strand_neighborhood'] = mmseqs_clust['strand'] + mmseqs_clust['rep'] # each protein in neighborhoods will be reprented by its cluster representative

    neighborhood_name_prots = {}
    for neighborhood_name in neighborhood_grp: # loop through neighborhoods to for gLM inputs
        mmseqs_clust_sub = mmseqs_clust[mmseqs_clust['neighborhood_name'] == neighborhood_name].copy()
        # print("Size of mmseq_clust after filter by neighborhood_grp",len(mmseqs_clust_sub))
        mmseqs_clust_sub.sort_values(by='start',inplace=True)
        mmseqs_clust_sub.drop_duplicates(subset=['start'],inplace=True) # in case something duplicate rows were made w/ earlier mmseqs merge
        center = mmseqs_clust_sub.VF_center.iloc[0] + '!!!' + mmseqs_clust_sub.gff.iloc[0] 
        mmseqs_clust_sub.reset_index(drop=True,inplace=True)
        ind = np.flatnonzero([mmseqs_clust_sub['prot_gffname']==center]) # index of the VF center in the neighborhood used for identifying its embedding in gLM output
        if len(ind) > 1:
            logger.warning(f"Multiple VF centers with the same name found in neighborhood: {neighborhood_name} ... grabing first vf_center")

        # assert int(mmseqs_clust_sub.start.iloc[-1]) -  int(mmseqs_clust_sub.start.iloc[0]) < args.neighborhood_size + 30000 # want to make sure that everything is working right, so if neighborhoods are much larger than expected, someting's wrong
        
        neighborhood_name_prots[neighborhood_name] = [';'.join(list(mmseqs_clust_sub.strand_neighborhood)),ind[0]]

    df = pd.DataFrame.from_dict(neighborhood_name_prots).T
    df.columns=['neighborhood','VF_center_index']

    if args.from_vfdb:
        # want vf categories in the results so that downstream classification is easier
        neighborhood_res = neighborhood_res[neighborhood_res["query"] == query][['vf_name','vf_id','vf_subcategory','vf_category','vfdb_species','vfdb_genus']]
        df["vf_name"],df["vf_id"],df["vf_subcategory"],df["vf_category"],df["vfdb_species"],df["vfdb_genus"] = neighborhood_res.iloc[0][0],neighborhood_res.iloc[0][1],neighborhood_res.iloc[0][2],neighborhood_res.iloc[0][3],neighborhood_res.iloc[0][4],neighborhood_res.iloc[0][5]
    else:
        df["vf_name"],df["vf_id"],df["vf_subcategory"],df["vf_category"],df["vfdb_species"],df["vfdb_genus"] = "non_vf","non_vf","non_vf","non_vf","non_vf","non_vf"
    #df_name = f"{neighborhood_res.iloc[0][1].replace(' ','_').replace(')','').replace('(','').replace('/','|')}_{str(random.randint(1,10000))}_{neighborhood_name}"
    df_name = query
    df.to_csv(f"{args.out}{glm_input_dir}/{df_name}.tsv",sep='\t',header=False,mode="x")

    fasta = SeqIO.parse(f"{args.out}combined_fastas_clust_rep.fasta","fasta") 
    fasta = filter(lambda x: x.id in list(mmseqs_clust.rep),fasta)
    with open(f"{args.out}{glm_input_dir}/{df_name}.fasta","x") as handle: #tsv and fasta files should have the same name
        SeqIO.write(fasta, handle, "fasta")
    return 