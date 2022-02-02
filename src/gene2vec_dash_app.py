# imports
import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
import argparse
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from ete3 import NCBITaxa
from copy import deepcopy

# describe program
parser = argparse.ArgumentParser(description='An interactive dashboard for gene embeddings.')

# arguments
parser.add_argument('--figure-json',
                    dest="json",
                    type=str,
                    help='File path of the gene2vec embedding to be plotted.')


# parse args
args = parser.parse_args()

# user defined function
def go_basic(tax_ids=[9606]):
    """Returns the basic gene annotations given a tax-id."""
    # imports
    from goatools.obo_parser import GODag
    from goatools.anno.genetogo_reader import Gene2GoReader

    # build dag
    godag = GODag("../../GO/go-basic.obo")

    # grab human annotations
    objanno = Gene2GoReader("../../GO/gene2go", taxids=tax_ids)
    go2geneids = objanno.get_id2gos(namespace="BP", go2geneids=True)

    # filter godag to taxid
    return {k: godag[k] for k in go2geneids.keys()}, go2geneids

def extract_gene_info(fig) -> pd.DataFrame:

    # generate gene info data
    gene_info = np.array(fig.data[0].customdata)
    num_cols = gene_info.shape[1]

    # extract gene identifiers
    fig_cols = fig.data[0].hovertemplate.split('<br>')
    fig_cols = [label.split("=")[0] for label in fig_cols]
    gene_info_cols = fig_cols[-num_cols:]
    
    # generate gene info
    gene_info = pd.DataFrame(data=np.array(fig.data[0].customdata), columns=gene_info_cols)
    
    return gene_info    

if __name__=="__main__":

    # define colors
    active = 'rgba(226,255,0,1)'; inactive='rgba(10, 10, 10, 0.01)'

    # load fig from json
    fig = pio.read_json(args.json)
    fig.update_traces(marker=dict(color=inactive))

    # check for type of gene representation
    gene_info = extract_gene_info(fig)
    gene_info_ids = gene_info.filter(regex="Entrez ID|Gene Symbol|Ensembl ID")
    tf = (gene_info_ids != "NA").all(axis=0)
    gene_rep = gene_info_ids.columns[tf][0]

    # compute go annotations
    tax_ids = gene_info.loc[gene_info['Tax ID'] != 'NA', 'Tax ID'].dropna().unique().astype(int)
    godag ,go2geneids = go_basic(tax_ids=tax_ids)
    go2geneids = pd.Series(go2geneids)
    go2geneids = go2geneids.iloc[(-go2geneids.apply(len)).argsort().values]

    # load reactome pathways
    reactome = pd.read_csv("/s/b/rna_seq/pathways/NCBI2Reactome_All_Levels.txt",
                           delimiter="\t",
                           dtype=str,
                           header=None
                           )
    reactome.columns = ['Entrez ID', 'Reactome ID', 'url', 'Name', "TAS/EXP", "Species"]

    # get reactome ids of taxid
    ncbi = NCBITaxa()
    taxid2name = ncbi.get_taxid_translator(tax_ids)
    species = list(taxid2name.values())
    reactome = reactome.query(f"Species in {species}")
    reactome = reactome.groupby("Reactome ID").agg(pd.unique)

    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "24rem",
        "padding": "2rem 1rem",
    }

    DROPDOWN_STYLE = {
        "padding": "4rem 0rem",
    }

    GENE2VEC_STYLE = {
    "margin-left": "24rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "height": "60rem"
    }

    # modules
    goid =  html.Div(
        [
            html.H4("Gene Ontology", className="display-8"),
            html.Hr(),
            dcc.Dropdown(
                id="GOID",
                options=[{"label": go_id, "value": go_id} for go_id in go2geneids.index],
                ),
        ],
        style=DROPDOWN_STYLE,
    )

    rid =  html.Div(
        [
            html.H4("Reactome ID", className="display-8"),
            html.Hr(),
            dcc.Dropdown(
                id="RID",
                options=[{"label": rid, "value": rid} for rid in reactome.index],
                ),
        ],
        style=DROPDOWN_STYLE,
    )

    
    description = html.Div(
        [   
            html.H5("Description", className="display-8"),
            html.Hr(),
            dcc.Textarea(
                id='description',
                readOnly=True,
                value='',
                style={'width': '100%', 'height': 300, 'color': "#fff", 'background-color': "#222"},
            )
        ]
    )


    sidebar = html.Div(
        [
            html.H2("GeneView", className="display-8"),
            html.Hr(),
            goid,
            rid,
            description,
        ],
        style=SIDEBAR_STYLE,
    )

    gene2vec = dcc.Graph(
                id="gene2vec",
                figure=fig,
                style=GENE2VEC_STYLE,
                )

    # app layout
    app.layout = html.Div(
        [
            sidebar,
            gene2vec,
        ],
        className="dash-bootstrap",
    )

    # callbacks
    @app.callback(
        Output("gene2vec", "figure"),
        [Input("GOID", "value")],
        [Input("RID", "value")],
        [State("gene2vec", 'figure')])
    def show_genes(go_id, rid, fig_json):
        # debug
        #print(go_id); print(rid)
        
        # get context and object clicked
        ctx = dash.callback_context
        id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # init entrez
        entrez_ids = None

        # set figure
        fig = go.Figure(fig_json)

        # use go id
        if id == "GOID":
            if goid is not None:
                # locate entrez ids
                entrez_ids = list(go2geneids.loc[go_id])
            else:
                return

        # use reactome id
        elif id == "RID":
            if rid is not None:
                # locate entrez ids
                entrez_ids = reactome.loc[rid, 'Entrez ID']
        
        # set inactive color
        gene_info['color'] = inactive

        if go_id is not None or rid is not None:
            tf = gene_info['Entrez ID'].isin([str(eid) for eid in entrez_ids])

            # fill in colors
            gene_info.loc[tf, 'color'] = active
            color = gene_info['color'].values.tolist()
            fig.update_traces(marker=dict(color=color))

            return fig
            
        return dash.no_update

    @app.callback(
        Output("description", "value"),
        [Input("GOID", "value")],
        [Input("RID", "value")])
    def show_description(go_id, rid):
        """Print the description of the GO ID"""

        # get context and object clicked
        ctx = dash.callback_context
        id = ctx.triggered[0]['prop_id'].split('.')[0]

        # print go id info
        if id == "GOID":
            if go_id is not None:
                # grab info
                go_info =  godag[go_id]

                # grab entrez ids
                entrez_ids = list(go2geneids.loc[go_id])
                gene_ids = gene_info.loc[gene_info['Entrez ID'].isin([str(eid) for eid in entrez_ids]), gene_rep].tolist()

                desc = f"GO ID: {go_info.id}\nName: {go_info.name}\n"\
                       f"Namespace: {go_info.namespace}\nLevel: {go_info.level}\n"\
                       f"Depth: {go_info.depth}\n{gene_rep}: {', '.join(gene_ids)}"

                return desc

        # print reactome id info
        elif id =="RID":
            if rid is not None:

                # grab entrez ids
                entrez_ids = reactome.loc[rid, 'Entrez ID']
                gene_ids = gene_info.loc[gene_info['Entrez ID'].isin(entrez_ids), gene_rep].tolist()

                # grab info
                rid_info =  reactome.loc[rid]

                desc = f"Reactome ID: {rid_info.name}\nName: {rid_info.Name[0]}\n"\
                       f"Species: {rid_info.Species[0]}\nurl: {rid_info.url[0]}\n"\
                       f"{gene_rep}: {', '.join(gene_ids)}"

                return desc

        return dash.no_update

    app.run_server(debug=True)