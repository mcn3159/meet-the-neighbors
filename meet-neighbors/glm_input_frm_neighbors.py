import pandas as pd
from Bio import SeqIO

def get_glm_input(query,uniq_neighborhoods_d,neighborhood_res,mmseqs_clust,args):
    if neighborhood_res.columns[-1] == "vf_category":
        if query not in list(neighborhood_res.vf_id): # skip if the vf was filtered out b/c of lack of hits decided by args.min_hits
            return
    
    elif (neighborhood_res.columns[-1] != "vf_category") and (query not in list(neighborhood_res.vf_query)):
        return

    # only keep queries that made it through filtering in main.py to reduce computations    
    neighborhood_grp = uniq_neighborhoods_d[query]
    mmseqs_clust = mmseqs_clust[mmseqs_clust['neighborhood_name'].isin(neighborhood_grp)]

    neighborhood_name_prots = {}
    for neighborhood_name in set(mmseqs_clust.neighborhood_name):
        mmseqs_clust_sub = mmseqs_clust[mmseqs_clust['neighborhood_name'] == neighborhood_name]
        mmseqs_clust_sub.sort_values(by='start',inplace=True)
        # want to make sure that everything is working right, so if neighborhoods are much larger than expected, someting's wrong
        assert int(mmseqs_clust_sub.start.iloc[-1]) -  int(mmseqs_clust_sub.start.iloc[0]) < args.neighborhood_size + 10000 
        mmseqs_clust_sub['strand_neighborhood'] = mmseqs_clust_sub['strand'] + mmseqs_clust_sub['rep'] #this might cause screaming
        neighborhood_name_prots[neighborhood_name] = ';'.join(list(mmseqs_clust_sub.strand_neighborhood))

    df = pd.DataFrame.from_dict([neighborhood_name_prots]).T
    df.columns=['neighborhood']

    if neighborhood_res.columns[-1] == "vf_category":
        # want vf categories in the results so that downstream classification is easier
        neighborhood_res = neighborhood_res[neighborhood_res["vf_id"] == query][['vf_id','vf_subcategory','vf_category']]
        df["vf_id"],df["vf_subcategory"],df["vf_category"] = neighborhood_res.iloc[0][0],neighborhood_res.iloc[0][1],neighborhood_res.iloc[0][2]
    else:
        df["vf_id"],df["vf_subcategory"],df["vf_category"] = "non_vf","non_vf","non_vf"
    df.to_csv(f"{args.out}glm_inputs/{query}_{neighborhood_name}.tsv",sep='\t',header=False)

    fasta = SeqIO.parse(f"{args.out}combined_fastas_clust_rep.fasta","fasta") 
    fasta = filter(lambda x: x.id in mmseqs_clust.locus_tag,fasta)
    with open(f"{args.out}glm_inputs/{query}_{neighborhood_name}.fasta","w") as handle: #tsv and fasta files should have the same name
        SeqIO.write(fasta, handle, "fasta")
    return 