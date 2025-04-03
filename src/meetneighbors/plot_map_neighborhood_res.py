import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
import math
import argparse
import warnings
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.exceptions import DataConversionWarning
import numpy as np
from scipy.stats import entropy
import dask.dataframe as dd

def prep_cluster_tsv(mmseqs_res_dir,logger):

    #run for mmseqs clustered results
    #example res: '/Users/mn3159/bigpurple/data/pirontilab/Students/Madu/bigdreams_dl/neighborhood_call/neighbors_rand5k/neighbors_rand5k_clust_res.tsv'

    # if statement here is to allow this function to be able to handle an already read in dataframe
    if isinstance(mmseqs_res_dir,str):
        mmseqs = dd.read_csv(mmseqs_res_dir,sep='\t',names=['rep','locus_tag'])
    else:
        mmseqs = mmseqs_res_dir.copy() 
    

    mmseqs['VF_center'],mmseqs['gff'],mmseqs['seq_id'],mmseqs['locus_range'],mmseqs['start'], mmseqs['strand'] = mmseqs['locus_tag'].str.split('!!!').str[1],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[2],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[3],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[4],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[5],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[6]
    mmseqs['neighborhood_name'] = mmseqs['VF_center'] + '!!!' + mmseqs['gff'] + '!!!' + mmseqs['seq_id'] + '!!!' + mmseqs['locus_range'] #VF_center in non_vf calls are simply just the query hits
    
    if isinstance(mmseqs_res_dir,str):
        mmseqs = mmseqs.compute()
    cluster_names = {rep:f"Cluster_{i}" for i,rep in enumerate(set(list(mmseqs.rep)))} #can't list and loop mmseqs col with dask, so I have to compute first
    mmseqs['cluster'] = mmseqs['rep'].map(cluster_names)
    mmseqs['prot_gffname'] = mmseqs['locus_tag'].str.split('!!!').str[0] + '!!!' + mmseqs['gff']

    logger.debug(f"Size of mmseqs cluster results: {mmseqs.shape}")
    return mmseqs

def map_vfcenters_to_vfdb_annot(prepped_mmseqs_clust,mmseqs_search,vfdb,removed_neighborhoods,logger):
    # map query search hits to each target neighborhood in the cluster df
    # what if the same protein fasta has multiple proteins with the same name
    # why do I merge by prot_gffname instead of just the prot name? I think there's a good reason, but I can't remember and didnt take good enough nnotes
    
    mmseqs_search['gff_name'] = mmseqs_search.tset.str.split('_protein.faa').str[0]
    mmseqs_search['prot_gffname'] = mmseqs_search['target'] + '!!!' + mmseqs_search['gff_name']

    # check how many query had all their neighborhoods removed b/c the minimum neighborhood conditions were not met
    if removed_neighborhoods != None:
        surviving_queries = len(set(mmseqs_search[~mmseqs_search['prot_gffname'].isin(removed_neighborhoods)]['query']))
        logger.debug(f"Number of queries that didn't pass minimum neighborhood criteria: {len(set(mmseqs_search['query'])) - surviving_queries} out of {len(set(mmseqs_search['query']))} with hits")

    mmseqs_search.drop_duplicates(subset=['prot_gffname'],inplace=True) # to reduce mmseqs_clust shape explosion

    # merge with how=left to keep NAs for proteins that part of the neighborhood but not a VF
    if vfdb:
        mmseqs_clust = dd.merge(prepped_mmseqs_clust, mmseqs_search[['query','prot_gffname',"vf_name","vf_subcategory","vf_id","vf_category",'vfdb_species','vfdb_genus']],on='prot_gffname',how="left") 
    else:
        mmseqs_clust = dd.merge(prepped_mmseqs_clust, mmseqs_search[['query','prot_gffname']],on='prot_gffname',how="left")
    logger.info(f"Size post search and cluster results merge: {mmseqs_clust.shape}")
    return mmseqs_clust

def reduce_overlap(mmseqs_clust_sub,window):
    # reduce overlapping neighborhood given window
    # take representative (close to median) of neighborhoods who's start positions are within window
    mmseqs_clust_sub_copy = mmseqs_clust_sub[1].copy()

    # each neighborhood has a respective locus range, we're gonna make 2 new columns w/ it for clustering
    mmseqs_clust_sub_copy[['nn_start','nn_stop']] = mmseqs_clust_sub_copy['locus_range'].str.split('-',expand=True).astype(int)

    # cluster neighborhoods with complete b/c we want minimize neighborhood loss
    clustering = AgglomerativeClustering(linkage='complete', distance_threshold=window, n_clusters= None).fit(
        mmseqs_clust_sub_copy[['nn_start','nn_stop']].to_numpy())
    
    # combine cluster labels w/ where each neighborhood start stop was found, indices are the same
    clu_df = pd.concat([mmseqs_clust_sub_copy[['nn_start','nn_stop']],
                        pd.DataFrame(clustering.labels_,columns=['clu_label'],index=mmseqs_clust_sub_copy.index)],
                        axis=1)

    # want the neighborhood closest to the median b/c that's most representaive?
    # pd.loc op, pandas median will average if subset is even, get the value closest to average if so
    med_nn_starts = list(clu_df.loc[clu_df.groupby('clu_label')['nn_start'].apply(lambda s: (s - s.median()).abs().idxmin())]['nn_start'])

    # copy df rows weren't modified, inds are the same, let's get the inds of all the rows that match the criteria
    inds_to_keep = mmseqs_clust_sub_copy[mmseqs_clust_sub_copy['nn_start'].isin(med_nn_starts)].index

    # keeping mmseqs_clust_sub instead of copy b/c I don't want newly cereated columns
    return mmseqs_clust_sub[1].loc[inds_to_keep]


def select_multivf_neighborhoods(mmseqs_clust,loose_vf_search,logger):
    # get a list of neighborhoods containing more than one VF based off of an additional mmseqs search with looser search parameters than the previous one

    mmseqs_clust_sub =  mmseqs_clust.dropna(subset=['query'])

    og_hits = dict(zip(mmseqs_clust_sub['locus_tag'],mmseqs_clust_sub['query']))

    # target are neighborhood lcs, queries are VFs
    loose_vf_search = loose_vf_search[~loose_vf_search['target'].isin(og_hits.keys())] # can remove the locus tags in which we already know is a query based on an earlier .9 seqid and cov search 
    lc_query_map = dict(zip(loose_vf_search['target'],loose_vf_search['query'])) # this dictionary will be used to map additional locus tags (target) that had some sort of seq similarity to a vf (query)

    og_hits.update(lc_query_map) # combine new locus tag hits to a VF with the originals
    mmseqs_clust['loose_query'] = mmseqs_clust['locus_tag'].map(og_hits) # new column gives an idea of what neighborhoods have multiple VFs mapped to it

    print(mmseqs_clust.dropna(subset=['loose_query']).shape)
    print(mmseqs_clust.dropna(subset=['loose_query']).head())
    query_nn = mmseqs_clust.dropna(subset=['loose_query']).groupby('neighborhood_name')['loose_query'].apply(list).to_dict()
    multi_query_nn = [nn for nn in query_nn if len(query_nn[nn])>1] # get neighborhood names that contain multiple VFs for future reference

    logger.debug(f"Number of neighborhoods with mutiple VFs found: {len(multi_query_nn)} out of {len(query_nn)} total neighborhoods")
    return mmseqs_clust,multi_query_nn

def get_query_neighborhood_groups(mmseqs_clust,cluster_neighborhoods_by):
    # may cause the loss of some queries, which may have been filtered out in red_olp
    # func results in a dictionary where key is prot query and values are rows in neighborhoods belonging to THAT query
    # needed b/c some queries may be in another query's neighborhood, resulting in glm inputs with neighborhoods belonging to the wrong query
    # has not been tested with cluster_neighborhoods_by != query
    mmseqs_clust_sub = mmseqs_clust.dropna(subset=[cluster_neighborhoods_by])
    query_prot = mmseqs_clust_sub.groupby(cluster_neighborhoods_by)['prot_gffname'].apply(list).to_dict()
    nname_query = mmseqs_clust_sub.groupby('neighborhood_name')[cluster_neighborhoods_by].apply(list).to_dict()
    nname_query = {n:query for n in nname_query for query in nname_query[n] if '!!!'.join(n.split('!!!')[:2]) in query_prot[query]}
    mmseqs_clust_nolink_targ_query = mmseqs_clust.copy()
    mmseqs_clust_nolink_targ_query[cluster_neighborhoods_by] = mmseqs_clust.neighborhood_name.map(nname_query) # no link between target and alot of the query col values
    mmseqs_clust_nolink_groups = mmseqs_clust_nolink_targ_query.groupby(cluster_neighborhoods_by,dropna=True) # did this line and the above so that the below dict comp runs faster hopefully
    return mmseqs_clust_nolink_groups

class VF_neighborhoods:
    def __init__(self,logger,mmseqs_clust,query,dbscan_eps,dbscan_min):
        self.query_prot = query
        self.dbscan_eps,self.dbscan_min = dbscan_eps,dbscan_min
        self.cdhit_sub_piv = pd.pivot_table(mmseqs_clust, index='neighborhood_name', aggfunc='size', columns='cluster',
                                        fill_value=0)
        self.total_hits = len(self.cdhit_sub_piv)

    def create_dist_matrix(self):
        self.cdhit_sub_piv.drop_duplicates(inplace=True)
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
        dist_matrix = pairwise_distances(self.cdhit_sub_piv.to_numpy(),metric='jaccard') # Distances instead of similarity for easier clustering
        unique_hits = len(self.cdhit_sub_piv)
        return dist_matrix,unique_hits

    def calc_clusters(self): # DBSCAN was used for plotting, I don't use much anymore
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
        db_res = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min, metric='jaccard').fit(self.cdhit_sub_piv) # using jaccard instead of hamming, b/c jaccard doesn't take into acct 0s (check Jul 8 2024 notes)
        return len(set(db_res.labels_)) - (1 if -1 in db_res.labels_ else 0), list(db_res.labels_).count(-1)
    
    def neighborhood_freqs(self): # get neighborhood entropies
        unique_neighborhoods = self.cdhit_sub_piv.groupby(self.cdhit_sub_piv.columns.to_list(),as_index=False).size() # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
        return entropy(unique_neighborhoods['size'].to_numpy())
    
    def get_neighborhood_names(self,threshold,linkage_method,logger):
        dist_matrix,unique_hits = self.create_dist_matrix()
        if len(dist_matrix) == 1: # don't cluster with only one neighborhood
            return list(self.cdhit_sub_piv.index)
        if threshold == 0: # for speed
            return list(self.cdhit_sub_piv.index)
        # goal of clustering is to find really similar neighborhoods, those with a jaccard distance below a threshold
        neighborhood_clusters = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold,affinity="precomputed",linkage=linkage_method).fit(dist_matrix) # sklearn version 1.1.2 is older so keyword metric is replaced with affinity
        unique_clusters = np.unique(neighborhood_clusters.labels_)
        rep_neighborhood_indices = []
        for c in unique_clusters:
            clust_inds = np.where(neighborhood_clusters.labels_==c)[0] # get indices for cluster members
            # no need to compute centroid if the cluster is just one neighborhood
            if clust_inds.shape[0] == 1: # if the cluster just has one member, no need to find centroid
                rep_neighborhood_indices.append(clust_inds[0])
            else:
                # want to find the centroid for clust members not whole matrix
                # print("Clust inds with multiple members:",clust_inds)
                dist_matrix_sub = dist_matrix[np.ix_(list(clust_inds),list(clust_inds))] # np.ix_ allows for indexing of matrices at row and col level, want centroid of clu members at row and col level
                centroid = np.mean(dist_matrix_sub, axis=1) # get the average distances
                distances_to_centroid = np.sum((dist_matrix_sub - centroid[:, np.newaxis]) ** 2, axis=1) # compute the euclidean distance to centroid, np.newaxis shapes centroid to allow for subtraction
                index_closest_to_centroid = np.argmin(distances_to_centroid) # get the index closest to centroid
                rep_neighborhood_indices.append(clust_inds[index_closest_to_centroid])
        cdhit_sub_piv_sub = self.cdhit_sub_piv.iloc[rep_neighborhood_indices,:] # want centroid neighborhoods, subset og df for this
        return list(cdhit_sub_piv_sub.index) # indicies are neighborhood names

    def to_dict(self):
        self.entropy = self.neighborhood_freqs() # want to calculate the entropy BEFORE create_dist_matrix() b/c that's where duplicates are removed
        self.dist_matrix, self.unique_hits = self.create_dist_matrix()
        self.clusters, self.noise = self.calc_clusters()
        return {
            "query" : self.query_prot,
            "total_hits" : self.total_hits,
            "unique_hits" : self.unique_hits,
            "clusters" : self.clusters,
            "noise" : self.noise,
            "entropy" : self.entropy
        }

def plt_neighborhoods(neighborhood_plt_df,out,vfdb):
    #hovering over bubbles may show same type of vf but each bubble is a diff vf query
    #alot of this code is from: https://stackoverflow.com/questions/71694358/bubble-size-legend-with-python-plotly

    if vfdb: 
        fig1 = px.scatter(neighborhood_plt_df, x="unique_hits", y="total_hits",
                    size="bubble_size", color="vf_category",log_x=True,log_y=True,
                        hover_name="vf_subcategory")
    else: 
        fig1 = px.scatter(neighborhood_plt_df, x="unique_hits", y="total_hits",
                    size="bubble_size",log_x=True,log_y=True,
                        hover_name="query")

    sizeref = 2.*max(neighborhood_plt_df['bubble_size'])/(80**2)

    fig1.update_traces(mode='markers', marker=dict(sizemode='area',
                                                sizeref=sizeref))

    df_l = neighborhood_plt_df.sort_values("bubble_size")
    fig2 = px.scatter(
        df_l,
        x=np.zeros(len(neighborhood_plt_df)),
        y=pd.qcut(df_l["bubble_size"], q=5, precision=0,duplicates='drop').astype(str),
        size="bubble_size"
    )

    fig2.update_traces(mode='markers', marker=dict(sizemode='area',
                                                sizeref=sizeref))


    fig = go.Figure(
        data=[t for t in fig1.data] + [t.update(xaxis="x2", yaxis="y2") for t in fig2.data],
        layout=fig1.layout,
    )

    # now config axes appropriately
    # need to adjust xaxis2 domain for future plots, especially after we filter out anything w/ less than 10 hits
    fig.update_layout(
        xaxis_domain=[0, 0.90],
        xaxis2={"domain": [0.90, 1], "matches": None, "visible": False},
        yaxis2={"anchor": "free", "overlaying": "y", "side": "right", "position": 1},
        showlegend=True,legend_x=1.2,title_text='Clustered Neighborhood Hits'
    )

    fig.write_html(f"{out}Bubbles_chart.html")
    return

def plt_hist_neighborh_clusts(neighborhood_plt_df,out):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=neighborhood_plt_df['clusters'],name='clusters'))
    fig.add_trace(go.Histogram(x=neighborhood_plt_df['noise'],name='noise'))

    # The two histograms are drawn on top of another
    fig.update_layout(barmode='stack',title_text='Distribution of Neighborhood clusters',
                    yaxis_title_text='Count',xaxis_title_text='Value')
    fig.update_xaxes(range=[0, int(max(list(neighborhood_plt_df['clusters']) + list(neighborhood_plt_df['noise'])))])
    fig.write_html(f"{out}Hist_clusters.html")
    return

def plt_regline_scatter(neighborhood_plt_df,out):
    fig = px.scatter(neighborhood_plt_df, x="unique_hits", y="total_hits", log_x=True, log_y=True, 
                 trendline="ols", trendline_options=dict(log_x=True,log_y=True),
                 title="Log-transformed Fit of Neighborhood Queries")
    results = px.get_trendline_results(fig)
    slope = str(results.iloc[0]["px_fit_results"].params[1])[:5] # https://stackoverflow.com/questions/63341840/plotly-how-to-find-coefficient-of-trendline-in-plotly-express
    fig.write_html(f"{out}scatter_trendline_{slope}_slope.html")
    return results

def plt_box_entropy(neighborhood_plt_df,out,vfdb):
    if vfdb:
        x = "vf_subcategory"
    else:
        x = "query"
    fig = px.box(neighborhood_plt_df,x=x,y="entropy",points='all',
                 title=f"Entropies across {x}")
    fig.write_html(f"{out}entropies_on_{x}.html")
    return

