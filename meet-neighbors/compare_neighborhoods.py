import numpy as np
import pandas as pd
from matplotlib import pyplot
import plotly.express as px
import plotly.graph_objects as go

# Functions in this file only work to compare 2 neighborhoods, I can edit to work with multiple neighborhoods if needed
def compare_neighborhood_entropy(*args,**kwargs):
    # args should be 2 neighborhood groups
    # if comparing vfs in rando proteins, should organize neighborhood df results by query instead of vf_id
    entropies1 = args[0].entropy
    entropies2 = args[1].entropy
    label1 = kwargs.get("label1","Group1")
    label2 = kwargs.get("label2","Group2")
    out = kwargs.get("out","")

    bins = np.linspace(0,max([max(entropies1),max(entropies2)]),50)

    pyplot.hist(entropies1, bins, alpha=0.5, label=label1,edgecolor='black',density=True)
    pyplot.hist(entropies2, bins, alpha=0.5, label=label2,edgecolor='black',density=True)
    pyplot.legend(loc='upper right')
    pyplot.ylabel('Frequency')
    pyplot.title("Histograms of Neighborhood Entropies Overalyed")
    pyplot.savefig(f"{out}Histogram_entropies.png")
    #pyplot.show()
    return 
#compare_neighborhood_entropy(neighborhood_plt_df,neighborhood_randos_plt_df,label1='VFs',label2='Random_gene')

def compare_uniqhits_trends(*args,**kwargs):
    neighborhood1 = args[0]
    neighborhood2 = args[1]
    label1 = kwargs.get("label1","Group1")
    label2 = kwargs.get("label2","Group2")
    out = kwargs.get("out","")
    write_table = kwargs.get("write_table",None)

    neighborhood1['Group'],neighborhood2['Group'] = label1,label2
    concat_neighborhoods = pd.concat([neighborhood1,neighborhood2],axis=0,ignore_index=True)

    fig = px.scatter(concat_neighborhoods, x="total_hits", y="unique_hits", color="Group", trendline="ols")
    if write_table:
        results = px.get_trendline_results(fig)
        f = open(f"{results.Group.iloc[0]}_regression_results.txt","w")
        print(results.px_fit_results.iloc[0].summary(),file=f)
        f.close()

        f = open(f"{results.Group.iloc[1]}_regression_results.txt","w")
        print(results.px_fit_results.iloc[1].summary(),file=f)
        f.close()
    fig.write_html(f"{out}Multiple_trendlines_scatter.html")
    return


