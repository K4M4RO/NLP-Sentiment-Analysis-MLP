import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os
import joblib

# Import de notre IA
from composants.ia_notes import NotesPredicteur

# -----------------------------------------------------------------------------
# INITIALISATION DU MODÈLE
# -----------------------------------------------------------------------------
print("Initialisation du modèle pour l'application Dash (Design Refondu)...")
predicteur = NotesPredicteur()
predicteur.charger_cerveau() # Charge le meilleur modèle sélectionné (Early Stopping)

# Variables pour Performance
try:
    r2_score = predicteur._score_ia.get('R2', "N/A")
    architecture = predicteur._mlp.hidden_layer_sizes
    
    # Métriques statiques simulées (ou basées sur votre dernier print) 
    f1_score_static = 0.81
    matrice_confusion = "[[23410  4580]\n [ 3912 28430]]"
except:
    r2_score = "N/A"
    architecture = "N/A"
    f1_score_static = "N/A"
    matrice_confusion = "N/A"

# -----------------------------------------------------------------------------
# STYLES CSS PERSONNALISÉS (SaaS Moderne)
# -----------------------------------------------------------------------------
CARD_STYLE = {
    "border-radius": "10px",
    "box-shadow": "0 4px 12px 0 rgba(0, 0, 0, 0.2)",
    "background-color": "#2d2d2d",
    "border": "1px solid #333",
    "margin-bottom": "20px"
}

TAB_STYLE = {
    "borderBottom": "1px solid #444",
    "borderTop": "none",
    "borderLeft": "none",
    "borderRight": "none",
    "padding": "12px",
    "fontWeight": "bold",
    "color": "#aaa",
    "backgroundColor": "transparent"
}

SELECTED_TAB_STYLE = {
    "borderTop": "none",
    "borderLeft": "none",
    "borderRight": "none",
    "borderBottom": "3px solid #3498db", 
    "backgroundColor": "transparent",
    "color": "#fff",
    "padding": "12px",
    "fontWeight": "bold"
}

# -----------------------------------------------------------------------------
# PRÉPARATION DES DONNÉES UMAP (Global)
# -----------------------------------------------------------------------------
chemin_umap = "data/umap_coords_dash.npy"
chemin_y = "data/umap_y_vrai_dash.npy"
chemin_csv = "data/books_rating_500k_filtre.csv"
df_umap_global = None

if os.path.exists(chemin_umap) and os.path.exists(chemin_y):
    coords = np.load(chemin_umap)
    notes = np.load(chemin_y)
    
    # On charge le dataframe globalement UNE FOIS au démarrage
    df_umap_global = pd.DataFrame({
        'UMAP 1': coords[:, 0],
        'UMAP 2': coords[:, 1],
        'Note': notes.ravel()
    })

def get_umap_figure(range_notes=[1, 5]):
    if df_umap_global is not None:
        # 1. Filtrage dynamique selon le slider
        mask = (df_umap_global['Note'] >= range_notes[0]) & (df_umap_global['Note'] <= range_notes[1])
        df_filtered = df_umap_global[mask]
        
        # 2. Sous-échantillonnage strict à 3000 points
        n_samples = min(3000, len(df_filtered))
        if n_samples > 0:
            df_plot = df_filtered.sample(n=n_samples, random_state=42)
        else:
            df_plot = df_filtered
        
        # 3. Chargement contextuel des textes d'avis (HOVER)
        if os.path.exists(chemin_csv) and len(df_plot) > 0:
            try:
                indices_echantillon = set(df_plot.index.tolist())
                
                # Astuce Pandas : skiprows filtre à la volée. 
                # (Ligne 0 est l'en-tête, les index Pandas 0..N correspondent aux lignes 1..N+1)
                skip_func = lambda x: (x - 1) not in indices_echantillon and x != 0
                df_textes = pd.read_csv(chemin_csv, skiprows=skip_func)
                
                col_texte = "review/text" if "review/text" in df_textes.columns else df_textes.columns[0]
                
                # On force les index pour s'aligner parfaitement lors de la fusion
                df_textes.index = sorted(list(indices_echantillon))
                df_plot = df_plot.join(df_textes[[col_texte]])
                
                # Troncature du texte à 100 caractères
                df_plot['Avis'] = df_plot[col_texte].astype(str).str[:100] + "..."
            except Exception as e:
                print("Avertissement Info Bulle:", e)
                df_plot['Avis'] = "Texte non disponible"
        else:
            df_plot['Avis'] = "Texte non disponible"

        # 4. Tracé avec Plasma et nouveau hover contextuel
        fig = px.scatter(
            df_plot, x='UMAP 1', y='UMAP 2',
            color='Note', 
            color_continuous_scale=px.colors.sequential.Plasma,
            title=f"Cartographie Sémantique ({len(df_plot)} avis échantillonnés)",
            hover_data={'UMAP 1': False, 'UMAP 2': False, 'Note': True, 'Avis': True}
        )
        
        # 5. Optimisation visuelle brillante (taille: 5, opacité: 0.8)
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        
        # 6. NETTOYAGE DU GRAPHIQUE
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=0, r=0, t=40, b=0),
            transition_duration=500, # Animation fluide quand le slider change
            coloraxis_colorbar=dict(
                title=dict(
                    text="Note",
                    side="top",
                    font=dict(color="#eee", size=14)
                ),
                thickness=12,
                len=0.8,
                yanchor="middle",
                y=0.5,
                tickfont=dict(color="#aaa")
            )
        )
        return fig
    else:
        fig = px.scatter(title="Veuillez générer l'UMAP depuis le CLI")
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig

# -----------------------------------------------------------------------------
# INTERFACE GRAPHIQUE DASH
# -----------------------------------------------------------------------------
# Ajout de Google Fonts Inter
font_url = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, font_url])
app.title = "NLP Dashboard - Prédiction d'Avis"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: 'Inter', sans-serif; }
            .rc-slider-tooltip-inner { background-color: #333 !important; color: #fff !important; }
            .nav-tabs .nav-link { color: #888; font-weight: 600; border: none; border-bottom: 2px solid transparent; }
            .nav-tabs .nav-link.active { color: #ffffff; background: transparent; border: none; border-bottom: 3px solid #eeeeee; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Contenu de l'Interface de Test (Désormais injectée dans un Tab)
tab1_content = dbc.CardBody([
    html.H5("Analyse de Sentiment Textuelle", className="card-title text-light mb-3", style={"fontWeight": "600"}),
    html.P("Saisissez un commentaire ci-dessous pour l'évaluer via le réseau neuronal.", className="text-muted"),
    dbc.Textarea(
        id="input-texte",
        placeholder="Ex: The cinematography was brilliant, but the pacing felt a bit off...",
        style={'height': '150px', 'backgroundColor': '#1a1a1a', 'color': '#fff', 'border': '1px solid #444', 'borderRadius': '8px'}
    ),
    html.Br(),
    dbc.Button("Évaluer le texte", id="btn-predire", className="w-100", style={"backgroundColor": "#444", "color": "white", "border": "1px solid #666", "borderRadius": "8px", "fontWeight": "600"}),
    html.Br(), html.Br(),
    html.Div(id="output-prediction")
])

# Contenu de la Cartographie Sémantique
tab2_content = dbc.CardBody([
    html.H5("Projection Sémantique Globale", className="card-title text-light mb-3", style={"fontWeight": "600"}),
    html.P("Topologie non-linéaire générée par l'algorithme UMAP optimisé avec Cosine Similarity. "
           "Cette représentation en 2D des embeddings met en évidence la séparabilité latente des avis structurée par le modèle BERT.", 
           className="text-muted", style={"fontSize": "0.9rem"}),
           
    # Le graphique + Activation de l'outillage de Zoom/Box (displayModeBar: True)
    dcc.Graph(id='umap-graph', figure=get_umap_figure(), config={'displayModeBar': True}),
    
    html.Div([
        html.H6("Filtrer par tranche de notes :", className="text-light mt-4 mb-3", style={"fontSize": "0.9rem"}),
        dcc.RangeSlider(
            id='slider-note-umap',
            min=1,
            max=5,
            step=0.5,
            value=[1, 5],
            marks={i: {'label': str(i), 'style': {'color': '#aaa'}} for i in range(1, 6)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={
        "padding": "15px 20px", 
        "backgroundColor": "#222", 
        "borderRadius": "8px", 
        "marginTop": "20px", 
        "border": "1px solid #333"
    })
])

# Contenu du Rapport de Performance
tab3_content = dbc.CardBody([
    html.H5("Métriques d'Évaluation du Modèle", className="mb-4 text-light", style={"fontWeight": "600"}),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("ARCHITECTURE MLP", className="text-muted mb-1", style={"fontSize": "0.8rem", "letterSpacing": "1px"}),
                html.H3(str(architecture), className="text-info", style={"fontWeight": "600"})
            ])
        ], style=CARD_STYLE)),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("R² SCORE (TEST)", className="text-muted mb-1", style={"fontSize": "0.8rem", "letterSpacing": "1px"}),
                html.H3(f"{r2_score:.4f}" if isinstance(r2_score, float) else r2_score, style={"color": "#2ecc71", "fontWeight": "600"})
            ])
        ], style=CARD_STYLE)),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("F1-SCORE CLASSIFICATION", className="text-muted mb-1", style={"fontSize": "0.8rem", "letterSpacing": "1px"}),
                html.H3(f"{f1_score_static}", style={"color": "#f1c40f", "fontWeight": "600"})
            ])
        ], style=CARD_STYLE))
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H6("MATRICE DE CONFUSION (SEUIL > 2.5)", className="text-muted mb-3", style={"fontSize": "0.8rem", "letterSpacing": "1px"}),
            html.Pre(matrice_confusion, style={
                "backgroundColor": "#1a1a1a", 
                "padding": "20px", 
                "borderRadius": "8px",
                "color": "#ecf0f1",
                "fontSize": "1.1rem",
                "border": "1px solid #444",
                "fontFamily": "monospace"
            })
        ])
    ])
])

app.layout = dbc.Container([
    # En-tête
    html.Div([
        html.H2("NLP Analytics Dashboard", className="text-light mb-1", style={"fontWeight": "600", "letterSpacing": "-0.5px"}),
        html.P("Analyse de Sentiment & Cartographie Sémantique de Textes", className="text-muted", style={"fontSize": "1rem"})
    ], className="mt-5 mb-5", style={"borderBottom": "1px solid #333", "paddingBottom": "20px"}),
    
    dcc.Tabs([
        dcc.Tab(label="Interface de Test", children=[dbc.Card([tab1_content], style=CARD_STYLE, className="mt-4")], style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
        dcc.Tab(label="Cartographie Sémantique UMAP", children=[dbc.Card([tab2_content], style=CARD_STYLE, className="mt-4")], style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
        dcc.Tab(label="Rapport de Performance", children=[dbc.Card([tab3_content], style=CARD_STYLE, className="mt-4")], style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE)
    ], style={"borderBottom": "1px solid #444"})
    
], fluid=False, style={"maxWidth": "1200px"})

# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------
@app.callback(
    Output("output-prediction", "children"),
    Input("btn-predire", "n_clicks"),
    State("input-texte", "value")
)
def update_prediction(n_clicks, texte):
    if not n_clicks or not texte:
        return dash.no_update
    
    try:
        score = predicteur.predire_score(texte)
        is_interessant = score > 2.5
        texte_resultat = "Intéressant" if is_interessant else "Inintéressant"
        
        couleur_bord = "#2ecc71" if is_interessant else "#e74c3c"
        couleur_texte = "#2ecc71" if is_interessant else "#e74c3c"
        
        return dbc.Card([
            dbc.CardBody([
                html.H6("RÉSULTAT DE L'INFÉRENCE", className="text-muted mb-2", style={"fontSize": "0.75rem", "letterSpacing": "1px"}),
                html.Div([
                    html.Div([
                        html.Span("Note estimée : ", style={"color": "#ccc"}),
                        html.Strong(f"{score:.2f} / 5", style={"color": "#fff", "fontSize": "1.2rem", "marginLeft": "10px"})
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Classification : ", style={"color": "#ccc"}),
                        html.Strong(texte_resultat, style={"color": couleur_texte, "fontSize": "1.1rem", "marginLeft": "10px"})
                    ])
                ])
            ])
        ], style={
            "backgroundColor": "#1e1e1e", 
            "border": "none", 
            "borderLeft": f"4px solid {couleur_bord}",
            "borderRadius": "4px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"
        })
        
    except Exception as e:
        return html.Div(f"Erreur système : {str(e)}", style={"color": "#e74c3c", "padding": "10px", "borderLeft": "3px solid #e74c3c", "backgroundColor": "#2a1515", "borderRadius": "4px"})

@app.callback(
    Output("umap-graph", "figure"),
    Input("slider-note-umap", "value")
)
def update_umap_graph(range_val):
    if range_val is None:
        return dash.no_update
    return get_umap_figure(range_val)

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Démarrage du serveur Dash localement sur http://127.0.0.1:8050/")
    app.run(debug=True)
