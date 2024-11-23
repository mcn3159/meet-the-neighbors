import pandas as pd
import numpy as np
import glob
import os
import pickle as pkl
from tqdm import tqdm


def unpack_embeddings(glm_out_dir,glm_in_dir,mmseqs_clust): # lowkey don't need mmseqs_clust but it's here just to make sure things are working
    glm_folders_o = [folder for folder in glob.glob(glm_out_dir+"*") if os.path.isdir(folder)]
    glm_vf_fldrs_o = {fold.split('/')[-1]:fold for fold in glm_folders_o}
    
    glm_folders_i = [folder for folder in glob.glob(glm_in_dir+"*.tsv")]
    glm_vf_fldrs_i = {fold.split('/')[-1].split('.ts')[0]:fold for fold in glm_folders_i}

    glm_res_d_vals_predf = {}
    # below column names are used for virulent mode, 
    input_tsv_cols = ['neighborhood_name','neighborhood','vf_center_index','vf_name','vfid','vf_subcategory','vf_category','species','genus']
    
    for vf in tqdm(glm_vf_fldrs_o):
        input_tsv = pd.read_csv(glm_vf_fldrs_i[vf],sep="\t",names=input_tsv_cols) #read in gLM input
        batch_size = [len(row.neighborhood.split(';')) for row in input_tsv.itertuples()] # get the whole size of each neighborhood for later indexing for VF center
        glm_index = [int(input_tsv.vf_center_index.iloc[i]) + sum(batch_size[:i]) for i in range(len(batch_size))] # grabbing index of protein in gLM batch output obj
        input_tsv['glm_indicies'] = glm_index
        query_reps = list(set(mmseqs_clust[mmseqs_clust['query']==vf]['rep'])) # VF_center cluster representatives only used for assert
        glm_batch = pkl.load(open(glm_vf_fldrs_o[vf]+'/results/results/batch.pkl.glm.embs.pkl','rb'))
        glm_batch = np.array(glm_batch, dtype=object)[input_tsv['glm_indicies'].values] #subset gLM batch for indicies of VF centers
        assert len(glm_batch) == len(input_tsv),"gLM batch size subset and input tsv size do not match"
        for i,embed in enumerate(glm_batch): # there should be an embedding for each row in input_tsv or for each VF center
            glm_res_d_vals_predf[vf+"!!!"+str(i)] = np.append(embed[1],input_tsv.iloc[i].loc[['neighborhood_name','vf_name','vfid','vf_category','species','genus']].values) # used i to index input_tsv instead of 0 b/c neighborhood names are different for each row
            assert embed[0] in query_reps, f"Embedding {embed[0]} in gLM batch {vf} is not a rep of a target VF." # make sure 

    return glm_res_d_vals_predf

def get_glm_embeddf(glm_res_d_vals_predf): # format dictionary to an embedding df for uMAP plotting
    
    embedding_df = pd.DataFrame.from_dict(glm_res_d_vals_predf,orient='index')
    embedding_df.reset_index(names="query",inplace=True) # names parameter here only works on newer versions of pandas
    embedding_df["query"] = embedding_df["query"].str.split('!!!').str[0]
    cols_names = ['neighborhood_name','vf_name','vfid','vf_category','species','genus']
    embedding_df.columns.values[-len(cols_names):] = cols_names
    return embedding_df

def get_umapdf(embedding_df):
    import umap # i know it's weird but this package is a pain to install

    cols_names = ['neighborhood_name','vf_name','vfid','vf_category','species','genus']
    umapper = umap.UMAP().fit(embedding_df.iloc[:,1:-len(cols_names)].values)
    return umapper

def plt_baby(umapper,embedding_df_merge,**kwargs):
    outdir = kwargs.get("outdir")
    plt_name = kwargs.get("plt_name")
    legend = kwargs.get("legend")
    width = kwargs.get("width")
    label = kwargs.get("label")
    legend = kwargs.get("legend")
    import umap
    import umap.plot # may need to conda install the requirements

    if not os.path.isfile(f"{outdir}{plt_name}.tsv"):
        embedding_df_merge.to_csv(f"{outdir}{plt_name}.tsv",sep="\t",index=False,mode="x")

    p = umap.plot.points(umapper, labels=embedding_df_merge[label],show_legend=legend,width=width)
    p_obj = p.get_figure()
    assert not os.path.isfile(f"{outdir}{plt_name}.png"), "File with this name already exists"
    p_obj.savefig(f"{outdir}{plt_name}.png")
    return 
