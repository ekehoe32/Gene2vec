"""This script is used to plot a gene2vec embedding"""

# imports
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
import mygene
import math
import os

# describe program
parser = argparse.ArgumentParser(description='Plots an embedding of a gene2vec hidden layer.')

# arguments
parser.add_argument('--embedding',
                    type=str,
                    help='File path of the gene2vec embedding to be plotted.')
parser.add_argument('--out',
                    type=str,
                    help='File path of output plot.',
                    default=None)

parser.add_argument('--plot-title',
                    dest='plot_title',
                    type=str,
                    help='Custom title for plot.',
                    default=None)

parser.add_argument('--alg',
                    type=str,
                    choices=['umap', 'pca', 'mds', 'tsne'],
                    default='umap',
                    help='The dimension reduction algorithm to used to produce the embedding.')

parser.add_argument('--species',
                    default=9606,
                    help='Species name or taxid used to generate the gene embedding.')

parser.add_argument('--dim',
                    type=int,
                    default=2,
                    help='Dimension of the embedding.')

# parse args
args = parser.parse_args()

# user defined functions
def load_embedding(filename):
    geneList = list()
    vectorList = list()
    f = open(filename)
    for line in f:
        values = line.split()
        gene = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        geneList.append(gene)
        vectorList.append(vector)
    f.close()
    return np.asarray(vectorList), np.asarray(geneList)

def infer_gene_rep(x) -> str:
    # check for entrez id
    if type(x) == int:
        return 'Entrez ID'
    elif type(x) == str:
        # check for ensembl id
        if 'ENS' in x:
            return 'Ensembl ID'
        else:
            # default it gene symbol
            return 'Gene Symbol'

def query_gene_info(gene_ids, species=9606):
    # infer type of gene id
    gene_rep = infer_gene_rep(gene_ids[0].item())

    # build querying object
    mg = mygene.MyGeneInfo()

    # excute query based upon species and gene rep
    if gene_rep == "Gene Symbol":
        gene_info = mg.querymany(gene_ids, scopes='symbol', species=species, as_dataframe=True)
        gene_info = gene_info.groupby("symbol").agg(unique_non_null)
        gene_info["symbol"] = gene_info.index
        return gene_info
    elif gene_rep == "Entrez ID":
        gene_info = mg.querymany(gene_ids, scopes='entrezgene', species=species, as_dataframe=True)
        gene_info = gene_info.groupby("entrezgene").agg(unique_non_null)
        gene_info["entrezgene"] = gene_info.index
        return gene_info
    elif gene_rep == "Ensembl ID":
        gene_info = mg.getgenes(gene_ids, fields='name,symbol,entrezgene,taxid', as_dataframe=True)
        gene_info = gene_info.groupby("query").agg(unique_non_null)
        gene_info["query"] = gene_info.index
        return gene_info

def unique_non_null(x):

    # drop na entry and get unique values
    y = x.dropna().unique()

    if y.size == 1:
        return y.item()
    elif y.size == 0:
        return pd.NA
    else:
        return y

if __name__=="__main__":

    # load gene2vec embedding
    print("\nRunning:")
    print(f"\t[*] Loading the Gene2vec embedding: {os.path.abspath(args.embedding)}...")
    wv, vocabulary = load_embedding(args.embedding)
    print(f"\t\t- Number of Genes: {'{:,}'.format(vocabulary.size)}.")
    print(f"\t\t- Embedding Dimension: {wv.shape[1]}.")

    # find gene info
    print(f"\t[*] Querying NCBI for gene info...")
    gene_info = query_gene_info(vocabulary, args.species)

    # define dimension reduction algorithm
    if args.alg == 'umap':
        from umap import UMAP
        reduce = UMAP(n_components=args.dim)
    elif args.alg == 'pca':
        from sklearn.decomposition import PCA
        reduce = PCA(n_components=args.dim, whiten=True)

    # reduce dimension
    print(f"\t[*] Reducing the dimension of Gene2vec embedding with {args.alg.upper()}(dim={args.dim})...")
    wv_red = reduce.fit_transform(wv)

    # create dataframe for plotting
    gene_rep = infer_gene_rep(vocabulary[0].item())
    df = pd.DataFrame(index=vocabulary, data=wv_red)
    df.loc[gene_info.index.values, "Gene Symbol"] = gene_info['symbol']
    df.loc[gene_info.index.values, "Tax ID"] = gene_info['taxid'] 
    df.loc[gene_info.index.values, "Entrez ID"] = gene_info['entrezgene']
    df.loc[gene_info.index.values, "Name"] = gene_info['name']  
    if gene_rep == "Ensembl ID":
        df.loc[vocabulary, "Ensembl ID"] = vocabulary
    elif gene_rep == "Gene Symbol":
        df.loc[vocabulary, "Gene Symbol"] = vocabulary
    elif gene_rep == "Entrez ID":
        df.loc[vocabulary, "Entrez ID"] = vocabulary

    # replace na
    df.fillna('NA', inplace=True)

    # generate hover data
    hover_data = df.filter(regex="Symbol|ID|Name").columns
    hover_data = {col: True for col in hover_data}

    # format columns
    col_dict = {0: f'{args.alg.upper()} 1', 1: f'{args.alg.upper()} 2', 2: f'{args.alg.upper()} 3'}
    df.rename(columns=col_dict, inplace=True)

    # plot
    print("\t[*] Generating interactive plot via plotly...")
    if args.dim == 2:
        fig = px.scatter(df, x=col_dict[0], y=col_dict[1],
                         hover_data=hover_data,
                         #color_continuous_scale="RdBu",
                         #opacity=.7,
                         size_max=8)
        fig.update_traces(marker=dict(color='rgba(255, 255, 255, 0.1)'))

    if args.dim == 3:
        fig = px.scatter_3d(df, x=col_dict[0], y=col_dict[1], z=col_dict[2],
                            hover_data=hover_data,
                            #color_continuous_scale="RdBu",
                            #opacity=.7,
                            size_max=8)
        fig.update_traces(marker=dict(color='rgba(10, 10, 10, 0.01)'))

    # update plot layout
    if args.plot_title is None:
        args.plot_title = f"Gene2vec Embedding using {args.alg.upper()}"

    fig.update_layout(template='plotly_dark',
                      title=args.plot_title,
                      font=dict(size=18))

    # save to file
    if args.out is None:
        embedding_name = os.path.basename(args.embedding).rstrip('.txt')
        args.out = f"../figures/{embedding_name}_{args.alg}_{args.dim}.html"
    fig.write_html(args.out)
    fig.write_json(args.out.replace('.html', '.json'))
    print(f"\t[*] Plot saved to {os.path.abspath(args.out)}(.json).")
    print("Complete!\n")
