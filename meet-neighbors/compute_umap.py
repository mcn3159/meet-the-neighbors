import pandas as pd
import numpy as np
import glob
import os
import pickle as pkl
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot # may need to conda install the requirements

def unpack_embeddings(glm_out_dir,mmseqs_clust):
    # out_dir = "/Users/mn3159/bigpurple/data/pirontilab/Students/Madu/bigdreams_dl/test_neighborhoods/vfdb_setb_allab_neighbors_byquery/glm_output_wip"
    glm_folders = [folder for folder in glob.glob(glm_out_dir+"*") if os.path.isdir(folder)]
    glm_vf_fldrs = {fold.split('/')[-1]:fold for fold in glm_folders}

    glm_res_d_vals_predf = {}
    mmseqs_clust["prot_ids"] = mmseqs_clust["locus_tag"].str.split("!!!").str[0]
    for vf in glm_vf_fldrs: # this loop a lil slow but I think the clarity is really important here
        mmseqs_clust_sub = mmseqs_clust[mmseqs_clust["query"]==vf]
        batch = pkl.load(open(glm_vf_fldrs[vf]+'/results/results/batch.pkl.glm.embs.pkl','rb'))
        for i,embed in enumerate(batch): #which one of the proteins in the representative cluster has a protein id that is the same as a VF_center
            mmseqs_clust_sub_sub = mmseqs_clust_sub[mmseqs_clust_sub["rep"]==embed[0]].copy()
            if mmseqs_clust_sub_sub["prot_ids"].isin(mmseqs_clust_sub_sub["VF_center"]).any():
                glm_res_d_vals_predf[vf+"!!!"+str(i)] = np.append(embed[1],embed[0]) # queries have multiple hits, adding an index to prevent key replacement when making dictionary
    embedding_df = pd.DataFrame.from_dict(glm_res_d_vals_predf,orient='index')
    embedding_df.reset_index(names="query",inplace=True)
    embedding_df["query"] = embedding_df["query"].str.split('!!!').str[0]
    return embedding_df

def get_glm_umap_df(embedding_df,mmseqs_clust):
    umapper = umap.UMAP().fit(embedding_df.iloc[:,1:-1].values)

    cols_to_add = ["query","vf_name","vf_subcategory","vf_id","vf_category","vfdb_genus","vfdb_species"]
    mmseqs_clust_for_merge = mmseqs_clust.drop_duplicates(subset=['query'])
    embedding_df_merge = pd.merge(embedding_df,mmseqs_clust_for_merge[cols_to_add],on='query')
    assert embedding_df_merge.shape[0] == embedding_df.shape[0], "Something went wrong with the previous merge"
    return umapper,embedding_df_merge

def plt_baby(umapper,embedding_df_merge,**kwargs):
    outdir = kwargs.get("outdir")
    plt_name = kwargs.get("plt_name")
    legend = kwargs.get("legend")
    width = kwargs.get("width")
    label = kwargs.get("label")
    legend = kwargs.get("legend")

    embedding_df_merge.to_csv(f"{outdir}{plt_name}.tsv",sep="\t",index=False,mode="x")

    p = umap.plot.points(umapper, labels=embedding_df_merge[label],show_legend=legend,width=width)
    p_obj = p.get_figure()
    assert not os.path.isfile(f"{outdir}{plt_name}.png"), "File with this name already exists"
    p_obj.savefig(f"{outdir}{plt_name}.png")
    return 
