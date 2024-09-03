import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    mmseqs = dd.read_csv(mmseqs_res_dir,sep='\t',names=['rep','locus_tag'])
    

    mmseqs['VF_center'],mmseqs['gff'],mmseqs['seq_id'],mmseqs['locus_range'],mmseqs['start'], mmseqs['strand'] = mmseqs['locus_tag'].str.split('!!!').str[1],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[2],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[3],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[4],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[5],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[6]
    mmseqs['neighborhood_name'] = mmseqs['VF_center'] + '!!!' + mmseqs['gff'] + '!!!' + mmseqs['seq_id'] + '!!!' + mmseqs['locus_range'] #VF_center in non_vf calls are simply just the query hits
    
    mmseqs = mmseqs.compute()
    cluster_names = {rep:f"Cluster_{i}" for i,rep in enumerate(set(list(mmseqs.rep)))} #can't list and loop mmseqs col with dask, so I have to compute first
    mmseqs['cluster'] = mmseqs['rep'].map(cluster_names)

    logger.info(f"Size of mmseqs cluster results: {mmseqs.shape}")
    return mmseqs

def map_vfcenters_to_vfdb_annot(prepped_mmseqs_clust,mmseqs_search,vfdb,logger):
    # map query search hits to each target neighborhood in the cluster df
    # what if the same protein fasta has multiple proteins with the same name
    prepped_mmseqs_clust['prot_gffname'] = prepped_mmseqs_clust['locus_tag'].str.split('!!!').str[0] + '!!!' + prepped_mmseqs_clust['gff']
    mmseqs_search['gff_name'] = mmseqs_search.tset.str.split('_protein.faa').str[0]
    mmseqs_search['prot_gffname'] = mmseqs_search['target'] + '!!!' + mmseqs_search['gff_name']
    mmseqs_search.drop_duplicates(subset=['prot_gffname'],inplace=True) # to reduce mmseqs_clust shape explosion
    if vfdb:
        mmseqs_clust = dd.merge(prepped_mmseqs_clust, mmseqs_search[['query','prot_gffname',"vf_name","vf_subcategory","vf_id","vf_category",'vfdb_species','vfdb_genus']],on='prot_gffname',how="left")
    else:
        mmseqs_clust = dd.merge(prepped_mmseqs_clust, mmseqs_search[['query','prot_gffname']],on='prot_gffname',how="left")
    logger.info(f"Head of merge results: {mmseqs_clust.head()}")
    return mmseqs_clust

def reduce_overlap(mmseqs_clust_sub,window):
    # reduce overlapping neighborhood given window
    # take representative (close to median) of neighborhoods who's start positions are within window
    similar_range_neighbors = {}
    for ran in set(mmseqs_clust_sub[1].locus_range):
        if len(similar_range_neighbors) == 0: 
            similar_range_neighbors[int(ran.split("-")[0])] = [ran]
            continue
        
        keys_array = np.array(list(similar_range_neighbors.keys())) # get an array of the neighborhood start positions
        keys_array_sub = np.absolute(keys_array - int(ran.split("-")[0])) < window # group new neigborhoods w/ keys in the same range
        res_ind = (keys_array_sub).nonzero() # get position of keys within the range of query neighborhood
        if np.size(res_ind) == 1:
            similar_range_neighbors[keys_array[res_ind[0]][0]].append(ran)
        
        else:
            similar_range_neighbors[int(ran.split("-")[0])] = [ran]

    similar_range_neighbors = {int(np.quantile([int(r.split("-")[0]) for r in v],q=.5,method="lower")):v for k,v in similar_range_neighbors.items()} # Want the overlapping neighborhood reps to be the middle neighborhood
    mmseqs_clust_sub_copy = mmseqs_clust_sub[1].copy()
    mmseqs_clust_sub_copy["neighborhood_start"] = mmseqs_clust_sub_copy["locus_range"].str.split("-").str[0]
    mmseqs_clust_sub_copy["neighborhood_start"] = pd.to_numeric(mmseqs_clust_sub_copy["neighborhood_start"])
    mmseqs_clust_sub_copy = mmseqs_clust_sub_copy[mmseqs_clust_sub_copy['neighborhood_start'].isin(list(similar_range_neighbors.keys()))]
    return mmseqs_clust_sub_copy

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
        db_res = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min, metric='jaccard').fit(self.cdhit_sub_piv)
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
        # picked single linkage b/c I should get the most unique/least amount of clusters, classification performance is good w/ the most unique neighborhoods
        neighborhood_clusters = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold,metric="precomputed",linkage=linkage_method).fit(dist_matrix) 
        unique_clusters = np.unique(neighborhood_clusters.labels_)
        rep_neighborhood_indices = []
        for c in unique_clusters:
            clust_inds = np.where(neighborhood_clusters.labels_==c)[0] # get indices for cluster members
            # no need to compute centroid if the cluster is just one neighborhood
            if clust_inds.shape[0] == 1: # if the cluster just has one member, no need to find centroid
                rep_neighborhood_indices.append(clust_inds[0])
            else:
                # want to find the centroid for clust members not whole matrix
                print("Clust inds with multiple members:",clust_inds)
                dist_matrix_sub = dist_matrix[np.ix_(list(clust_inds),list(clust_inds))] # np.ix_ allows for indexing of matrices at row and col level, want centroid of clu members at row and col level
                centroid = np.mean(dist_matrix_sub, axis=1) # get the average distances
                distances_to_centroid = np.sum((dist_matrix_sub - centroid[:, np.newaxis]) ** 2, axis=1) # compute the euclidean distance to centroid, np.newaxis shapes centroid to allow for subtraction
                index_closest_to_centroid = np.argmin(distances_to_centroid) # get the index closest to centroid
                rep_neighborhood_indices.append(clust_inds[index_closest_to_centroid])
        cdhit_sub_piv_sub = self.cdhit_sub_piv.iloc[rep_neighborhood_indices,:] # want centroid neighborhoods, subset og df for this
        return list(cdhit_sub_piv_sub.index) # indicies are neighborhood names

    def to_dict(self):
        self.dist_matrix, self.unique_hits = self.create_dist_matrix()
        self.clusters, self.noise = self.calc_clusters()
        self.entropy = self.neighborhood_freqs()
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

