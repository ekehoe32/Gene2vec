"""This script is used to generate gene pairs for the gene2vec model from a query"""

# imports
import argparse
import pandas as pd
import numpy as np
from copy import deepcopy
import os
import ray

# describe program
parser = argparse.ArgumentParser(description='Generate gene co-expression pairs from a processed query for a downstream gene2vec model.')

# arguments
parser.add_argument('--query', type=str, help='File path of the directory containing the query.')
parser.add_argument('--out',
                    type=str,
                    help='File path of output gene pairs.',
                    default="../data/gene_pairs.txt")
parser.add_argument('--corr-threshold',
                    type=float,
                    dest='corr_threshold',
                    default=.9,
                    help='Value to threshold correlation at for gene co-expression.')
parser.add_argument('--min-study-samples',
                    type=int,
                    dest='min_study_samples',
                    default=20,
                    help='Minimum number of samples which must be present in each study.')
parser.add_argument('--parallel',
                    dest='parallel',
                    action='store_true',
                    help='Indicates to use parallel processing via ray.')
parser.add_argument('--ensembl',
                    dest='ensembl',
                    action='store_true',
                    help='Indicates to use ensembl id over gene name.')

parser.set_defaults(parallel=False, ensembl=False)

# parse args
args = parser.parse_args()

# user defined functions
def coexpr(data: pd.DataFrame) -> list:

    # compute gene correlations
    print("Computing gene correlations...")
    corr = data.corr().abs()

    # get indices of highly-correlated
    rows, cols = (corr>args.corr_threshold).values.nonzero()

    # init output
    gene_pairs = []

    # loop through indices
    print(f"Computing gene co-expressions with correlation threshold={args.corr_threshold}...")
    for row, col in zip(rows, cols):

        if row != col:
            # extract gene identifiers
            gene_pairs.append(f"{corr.index[row]} {corr.index[col]}")
    
    return gene_pairs

def gene_name(gene_fields: list) -> str:
       try:
              return gene_fields[1]
       except IndexError:
              return ""

def half_min(x: np.ndarray) -> float:

    # find non-zero minimum
    y = x[x>0]
    hm = y.min()/2

    return hm

def clean_and_normalize(data: pd.DataFrame, gene_counts: pd.DataFrame, sample_ids: list = None) -> pd.DataFrame:

    # check sample ids
    if sample_ids is None:
        sample_ids = data.index.tolist()

    # reformat gene counts
    print("Computing low expression genes (total counts ≤ 10)...")
    gene_split = gene_counts["gene_id"].str.split("|")
    ensembl_ids = [g[0] for g in gene_split]
    gene_total_expr = pd.Series(index=ensembl_ids, data=gene_counts.loc[:, sample_ids].sum(axis=1).values)

    # remove under expressed genes
    print("Removing low expression genes...")
    data_normed = deepcopy(data.loc[sample_ids, gene_total_expr >= 10])

    # replace 0 with half-min
    print("Replacing 0 with non-zero half-minimum...")
    data_normed = data_normed.replace(0.0, half_min(data))
    print("log2 normalizing data...")
    data_normed = data_normed.apply(np.log2)

    return data_normed

def gene_annotated_data(data: pd.DataFrame, gene_counts: pd.DataFrame, sample_ids: list = None) -> pd.DataFrame:

    # load in data
    data_normed = clean_and_normalize(data, gene_counts, sample_ids)

    # load in gene names
    gene_split = gene_counts["gene_id"].str.split("|")
    gene_names = {g[0]: gene_name(g) for g in gene_split}

    # restrict data to genes which names
    print("Restricting data to genes with unique gene names...")
    data_normed.rename(columns=gene_names, inplace=True)
    has_gene_name = (data_normed.columns != "")
    data_normed = data_normed.loc[:, has_gene_name]

    # restrict data to genes which have unique names
    gene_name_counts = data_normed.columns.value_counts()
    unique_gene_names = gene_name_counts.index[(gene_name_counts == 1)]
    data_normed = data_normed.loc[:, unique_gene_names]

    return data_normed

@ray.remote
def generate_gene_name_pairs(data: pd.DataFrame, gene_counts: pd.DataFrame, sample_ids: list = None):

    # compute gene annotated data
    data_normed = gene_annotated_data(data, gene_counts, sample_ids)
    
    return coexpr(data_normed)

@ray.remote
def generate_gene_ensembl_pairs(data: pd.DataFrame, gene_counts: pd.DataFrame, sample_ids: list = None):

    # compute gene annotated data
    data_normed = clean_and_normalize(data, gene_counts, sample_ids)
    
    return coexpr(data_normed)

if __name__ == "__main__":
    print("\nRunning:")

    # load run table
    print("\t[*] Loading SRA Run Table...")
    query_dir = args.query
    run_table = pd.read_csv(os.path.join(query_dir, "data/SRARunTable.csv"), index_col=0)

    # load in data
    print("\t[*] Loading TPM data...")
    data = pd.read_csv(os.path.join(query_dir, "data/gene_counts_TPM.csv"), index_col=0)

    # load in gene counts
    print("\t[*] Loading gene counts for filtering...")
    gene_counts = pd.read_csv(os.path.join(query_dir, "data/gene_counts.csv"))

    # restrict data to run table
    data = data.loc[run_table.index.tolist()]

    # find studies
    study_counts = run_table["SRA Study"].value_counts()
    studies = study_counts.index[[study_counts >= args.min_study_samples]].tolist()

    # check for parallel execution
    if args.parallel:

        # start ray server
        ray.init()

        # put data and gene_counts in the store
        data_remote = ray.put(data)
        gene_counts_remote = ray.put(gene_counts)

        # loop through studies
        futures = []
        for study in studies:

            # compute sample ids
            sample_ids = run_table.index[(run_table["SRA Study"] == study)].tolist()
            if args.ensembl:
                futures.append(generate_gene_ensembl_pairs.remote(data_remote, gene_counts_remote, sample_ids))
            else:
                futures.append(generate_gene_name_pairs.remote(data_remote, gene_counts_remote, sample_ids))

        # gather results
        results = ray.get(futures)

        # shutdown ray
        ray.shutdown()
    else:
        # loop through studies
        results = []
        for study in studies:

            # compute sample ids
            sample_ids = run_table.index[(run_table["SRA Study"] == study)].tolist()
            if args.ensembl:
                results.append(coexpr(clean_and_normalize(data, gene_counts, sample_ids)))
            else:
                results.append(coexpr(gene_annotated_data(data, gene_counts, sample_ids)))

    # write gene pairs to file
    total_gene_pairs = sum([len(gene_pairs) for gene_pairs in results])
    print(f"\t[*] Writing gene pairs to file: {os.path.abspath(args.out)}...")
    with open(args.out, 'w+') as file:
        for gene_pairs in results:
            file.write('\n'.join(gene_pairs))
    print(f"\t[*] {'{:,}'.format(total_gene_pairs)} total co-expression gene pairs computed.")
    print("Complete!\n")