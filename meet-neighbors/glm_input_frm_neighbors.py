import pandas as pd
from Bio import SeqIO
import random 

def get_glm_input(query,uniq_neighborhoods_d,neighborhood_res,mmseqs_clust,args):
    pd.options.mode.chained_assignment = None
    if query not in list(neighborhood_res['query']): # skip if the vf was filtered out b/c of lack of hits decided by args.min_hits
        return

    # only keep queries that made it through filtering in main.py to reduce computations    
    neighborhood_grp = uniq_neighborhoods_d[query]
    mmseqs_clust = mmseqs_clust[mmseqs_clust['query'] == query]
    mmseqs_clust = mmseqs_clust[mmseqs_clust['neighborhood_name'].isin(neighborhood_grp)]

    mmseqs_clust['strand_neighborhood'] = mmseqs_clust['strand'] + mmseqs_clust['rep']


    neighborhood_name_prots = {}
    for neighborhood_name in set(mmseqs_clust.neighborhood_name):
        mmseqs_clust_sub = mmseqs_clust[mmseqs_clust['neighborhood_name'] == neighborhood_name]
        # print("Size of mmseq_clust after filter by neighborhood_grp",len(mmseqs_clust_sub))
        mmseqs_clust_sub.sort_values(by='start',inplace=True)
        
        # assert int(mmseqs_clust_sub.start.iloc[-1]) -  int(mmseqs_clust_sub.start.iloc[0]) < args.neighborhood_size + 30000 # want to make sure that everything is working right, so if neighborhoods are much larger than expected, someting's wrong
        
        neighborhood_name_prots[neighborhood_name] = ';'.join(list(mmseqs_clust_sub.strand_neighborhood))

    df = pd.DataFrame.from_dict([neighborhood_name_prots]).T
    df.columns=['neighborhood']

    if args.from_vfdb:
        # want vf categories in the results so that downstream classification is easier
        neighborhood_res = neighborhood_res[neighborhood_res["query"] == query][['vf_name','vf_id','vf_subcategory','vf_category','vfdb_species','vfdb_genus']]
        df["vf_name"],df["vf_id"],df["vf_subcategory"],df["vf_category"],df["vfdb_species"],df["vfdb_genus"] = neighborhood_res.iloc[0][0],neighborhood_res.iloc[0][1],neighborhood_res.iloc[0][2],neighborhood_res.iloc[0][3],neighborhood_res.iloc[0][4],neighborhood_res.iloc[0][5]
    else:
        df["vf_name"],df["vf_id"],df["vf_subcategory"],df["vf_category"],df["vfdb_species"],df["vfdb_genus"] = "non_vf","non_vf","non_vf","non_vf","non_vf","non_vf"
    #df_name = f"{neighborhood_res.iloc[0][1].replace(' ','_').replace(')','').replace('(','').replace('/','|')}_{str(random.randint(1,10000))}_{neighborhood_name}"
    df_name = query
    df.to_csv(f"{args.out}glm_inputs/{df_name}.tsv",sep='\t',header=False,mode="x")

    fasta = SeqIO.parse(f"{args.out}combined_fastas_clust_rep.fasta","fasta") 
    fasta = filter(lambda x: x.id in list(mmseqs_clust.rep),fasta)
    with open(f"{args.out}glm_inputs/{df_name}.fasta","x") as handle: #tsv and fasta files should have the same name
        SeqIO.write(fasta, handle, "fasta")
    return 