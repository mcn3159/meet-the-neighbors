from collections import defaultdict 
import pandas as pd
from Bio import SeqIO

def get_glm_input(query,uniq_neighborhoods_d,out_dir,neighborhood_res):
    if neighborhood_res.columns[-1] == "vf_category":
        if query not in list(neighborhood_res.vf_id): # skip if the vf was filtered out b/c of lack of hits decided by args.min_hits
            return
    
    elif (neighborhood_res.columns[-1] != "vf_category") and (query not in list(neighborhood_res.vf_query)):
        return
        
    fasta = SeqIO.parse(f"{out_dir}all_neighborhoods.fasta","fasta")
    neighborhood_grp = uniq_neighborhoods_d[query]
    fasta_filt = [(rec,int(rec.description.split("--start:")[1].split('--strand')[0])) for rec in fasta if rec.id.split('----',1)[1] in neighborhood_grp] # [(seq_record,position),...]
    fasta_filt = sorted(fasta_filt,key = lambda x:x[1]) #sorted because glm input cares about position of each protein
    neighborhood_name_prots = defaultdict(str)
    for rec in fasta_filt: # each neighborhood for each query sorted by name of neighborhood
        neighborhood_name = rec[0].id.split('----',1)[1]
        strand = rec[0].description.split("--strand:")[1]
        if len(neighborhood_name_prots[neighborhood_name]) != 0:
            neighborhood_name_prots[neighborhood_name] = neighborhood_name_prots[neighborhood_name] + ',' + strand + rec[0].id
        else:
            neighborhood_name_prots[neighborhood_name] = neighborhood_name_prots[neighborhood_name] + strand + rec[0].id
    df = pd.DataFrame.from_dict([neighborhood_name_prots]).T
    df.columns=['neighborhood']

    if neighborhood_res.columns[-1] == "vf_category":
        neighborhood_res = neighborhood_res[neighborhood_res["vf_id"] == query][['vf_id','vf_subcategory','vf_category']]
        df["vf_id"],df["vf_subcategory"],df["vf_category"] = neighborhood_res.iloc[0][0],neighborhood_res.iloc[0][1],neighborhood_res.iloc[0][2]
    else:
        df["vf_id"],df["vf_subcategory"],df["vf_category"] = "non_vf","non_vf","non_vf"
    df.to_csv(f"{out_dir}glm_inputs/{query}_{neighborhood_name}.tsv",sep='\t',header=True)

    with open(f"{out_dir}glm_inputs/{query}_{neighborhood_name}.fasta","w") as handle: #tsv and fasta files should have the same name
        SeqIO.write([rec_tuple[0] for rec_tuple in fasta_filt], handle, "fasta")
    return 