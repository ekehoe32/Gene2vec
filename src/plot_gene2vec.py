"""This script is used to plot a gene2vec embedding"""

# imports
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
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

if __name__=="__main__":

    # load gene2vec embedding
    print("\nRunning:")
    print(f"\t[*] Loading the Gene2vec embedding: {os.path.abspath(args.embedding)}...")
    wv, vocabulary = load_embedding(args.embedding)
    print(f"\t\t- Number of Genes: {'{:,}'.format(vocabulary.size)}.")
    print(f"\t\t- Embedding Dimension: {wv.shape[1]}.")

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
    df = pd.DataFrame(data=wv_red)
    df['Gene ID'] = vocabulary

    # format columns
    col_dict = {0: f'{args.alg.upper()} 1', 1: f'{args.alg.upper()} 2', 2: f'{args.alg.upper()} 3'}
    df.rename(columns=col_dict, inplace=True)

    # plot
    print("\t[*] Generating interactive plot via plotly...")
    if args.dim == 2:
        fig = px.scatter(df, x=col_dict[0], y=col_dict[1],
                         color=col_dict[1],
                         hover_data={'Gene ID': True},
                         color_continuous_scale="RdBu",
                         opacity=.7,
                         size_max=8)
    if args.dim == 3:
        fig = px.scatter_3d(df, x=col_dict[0], y=col_dict[1], z=col_dict[2],
                            color=col_dict[2],
                            hover_data={'Gene ID': True},
                            color_continuous_scale="RdBu",
                            opacity=.7,
                            size_max=8)

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
    print(f"\t[*] Plot saved to {os.path.abspath(args.out)}.")
    print("Complete!\n")
