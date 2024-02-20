import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import argparse
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.stats import entropy
import dask.dataframe as dd

class VF_neighborhoods:
    def __init__(self,cdhit_sub_vf,dbscan_eps,dbscan_min):
        self.VF_center = cdhit_sub_vf.iloc[0]['query'] # original protein used to search
        self.dbscan_eps,self.dbscan_min = dbscan_eps,dbscan_min
        self.cdhit_sub_piv = pd.pivot_table(cdhit_sub_vf, index='neighborhood_name', aggfunc='size', columns='cluster',
                                        fill_value=0)
        self.total_hits = len(self.cdhit_sub_piv)

    def create_dist_matrix(self):
        self.cdhit_sub_piv.drop_duplicates(inplace=True)
        dist_matrix = 1 - pairwise_distances(self.cdhit_sub_piv.to_numpy(),metric='hamming')
        unique_hits = len(self.cdhit_sub_piv)
        return dist_matrix,unique_hits

    def calc_clusters(self):
        db_res = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min, metric='hamming').fit(self.cdhit_sub_piv)
        return len(set(db_res.labels_)) - (1 if -1 in db_res.labels_ else 0), list(db_res.labels_).count(-1)
    
    def neighborhood_freqs(self):
        unique_neighborhoods = self.cdhit_sub_piv.groupby(self.cdhit_sub_piv.columns.to_list(),as_index=False).size() # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
        return entropy(unique_neighborhoods['size'].to_numpy())
    
    def get_neighborhood_names(self,threshold):
        dist_matrix,unique_hits = self.create_dist_matrix()
        lower_diag = np.tril(dist_matrix,k=-1) # grabs the lower half diagonal of distance matrix
        indices_to_remove = np.unique((lower_diag > threshold).nonzero()[0]) # get indices of lower half that contain values above threshold
        cdhit_sub_piv_sub = self.cdhit_sub_piv.drop(self.cdhit_sub_piv.iloc[list(indices_to_remove)].index) #remove those indices from dataframe
        return list(cdhit_sub_piv_sub.index)

    def to_dict(self):
        self.dist_matrix, self.unique_hits = self.create_dist_matrix()
        self.clusters, self.noise = self.calc_clusters()
        self.entropy = self.neighborhood_freqs()
        return {
            "vf_query" : self.VF_center,
            "total_hits" : self.total_hits,
            "unique_hits" : self.unique_hits,
            "clusters" : self.clusters,
            "noise" : self.noise,
            "entropy" : self.entropy
        }

def prep_mmseqs_tsv(mmseqs_res_dir):
    #run for mmseqs clustered results
    #example res: '/Users/mn3159/bigpurple/data/pirontilab/Students/Madu/bigdreams_dl/neighborhood_call/neighbors_rand5k/neighbors_rand5k_clust_res.tsv'
    mmseqs = dd.read_csv(mmseqs_res_dir,sep='\t',names=['rep','locus_tag'])
    

    mmseqs['VF_center'],mmseqs['gff'],mmseqs['seq_id'],mmseqs['locus_range'],mmseqs['start'], mmseqs['strand'] = mmseqs['locus_tag'].str.split('!!!').str[1],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[2],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[3],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[4],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[5],\
                                                                               mmseqs['locus_tag'].str.split('!!!').str[6]
    mmseqs['neighborhood_name'] = mmseqs['VF_center'] + '!!!' + mmseqs['gff'] + '!!!' + mmseqs['seq_id'] + '!!!' + mmseqs['locus_range']
    
    mmseqs = mmseqs.compute()
    cluster_names = {rep:f"Cluster_{i}" for i,rep in enumerate(set(list(mmseqs.rep)))} #can't list and loop mmseqs col with dask, so I have to compute first
    mmseqs['cluster'] = mmseqs['rep'].map(cluster_names)

    print(f"Size of mmseqs cluster results: {mmseqs.shape}")
    
    #mmseqs.head()
    return mmseqs

def map_vfcenters_to_vfdb_annot(prepped_mmseqs_clust,mmseqs_search,vfdb):
    # map query search hits to each target neighborhood in the cluster df
    # what if the same protein fasta has multiple proteins with the same name
    prepped_mmseqs_clust['vfname_gffname'] = prepped_mmseqs_clust['VF_center'] + '!!!' + prepped_mmseqs_clust['gff']
    mmseqs_search['gff_name'] = mmseqs_search.tset.str.split('_protein.faa').str[0]
    mmseqs_search['vfname_gffname'] = mmseqs_search['target'] + '!!!' + mmseqs_search['gff_name']
    if vfdb:
        mmseqs_clust = dd.merge(prepped_mmseqs_clust,
                            mmseqs_search[['query','vfname_gffname', 'vf_name', 'vf_subcategory', 'vf_id', 'vf_category']],
                            on='vfname_gffname')
    else:
        mmseqs_clust = dd.merge(prepped_mmseqs_clust, mmseqs_search[['query','vfname_gffname']],on='vfname_gffname')
    mmseqs_clust.head()
    return mmseqs_clust

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
    
def run():
    if __name__ == "__main__":
        print('running')
        parser = argparse.ArgumentParser(description="Meet-the-neighbors extracts genomic neighborhoods and runs analyses on them from protein fastas and their respective gffs")
        parser.add_argument("--mmseqs_tsv", type=str, required=True, default=None, help="Give mmseqs clustered tsv")
        parser.add_argument("--from_vfdb",required=False,action='store_true',help="Group vf centers of neighborhoods by their vf_id")
        parser.add_argument("--plot",required=False,action='store_true',help="Plot neighborhood cluster results")
        parser.add_argument("--out",required=True,type=True,help="Give output a name")
        args = parser.parse_args()

        # mmseqs_res_dir = '/Users/mn3159/bigpurple/data/pirontilab/Students/Madu/bigdreams_dl/neighborhood_call/neighbors_all_ids/neighbors_all_ids_clust_res.tsv'
        mmseqs_clust = prep_mmseqs_tsv(args.mmseqs_tsv)
        mmseqs_clust = map_vfcenters_to_vfdb_annot(mmseqs_clust,args.mmseqs_tsv,vfdb=args.from_vfdb)

        cluster_neighborhoods_by = "query"
        if args.from_vfdb:
            cluster_neighborhoods_by = "vf_id"
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        class_objs = {vf:VF_neighborhoods(cdhit_sub_vf=mmseqs_clust[mmseqs_clust[cluster_neighborhoods_by]==vf],dbscan_eps=0.15,dbscan_min=3)
                    for vf in set(mmseqs_clust[cluster_neighborhoods_by])}
        neighborhood_plt_df = pd.DataFrame.from_dict([class_objs[n].to_dict() for n in class_objs])
        neighborhood_plt_df = neighborhood_plt_df[neighborhood_plt_df['total_hits']>5]
        neighborhood_plt_df['bubble_size'] = (100/(neighborhood_plt_df['noise']+1)).astype(int)
        if args.from_vfdb:
            mmseqs_clust.rename(columns={'query':'vf_query'},inplace=True) #renaming here for clarity and mmseqs orginal tsv uses name query
            mmseqs_clust_for_merge = mmseqs_clust.drop_duplicates(subset=['vf_query'])
            neighborhood_plt_df = pd.merge(neighborhood_plt_df,mmseqs_clust_for_merge[['vf_query','vf_name','vf_id','vf_subcategory','vf_category']],on='vf_query',how='left')

        neighborhood_plt_df.to_csv(f"{args.out}_neighborhood_results_df.tsv",sep='\t',index=False,headers=True)
        plt_neighborhoods(neighborhood_plt_df,args.out)
        plt_hist_neighborh_clusts(neighborhood_plt_df,args.out)
        return
#run()
