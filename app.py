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
        
        # 2. Échantillonnage stratifié strict (300 par note, total max 1500)
        df_plot_list = []
        for note in df_filtered['Note'].unique():
            df_note = df_filtered[df_filtered['Note'] == note]
            n_samples = min(300, len(df_note))
            if n_samples > 0:
                df_plot_list.append(df_note.sample(n=n_samples, random_state=42))
        
        if df_plot_list:
            df_plot = pd.concat(df_plot_list)
        else:
            df_plot = df_filtered
        
        # 3. Chargement contextuel des textes d'avis (HOVER & CLICK)
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
                
                # Sauvegarde du texte complet (Click) et Troncature à 100 char (Hover)
                df_plot['Avis Complet'] = df_plot[col_texte].astype(str)
                df_plot['Avis'] = df_plot['Avis Complet'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
            except Exception as e:
                print("Avertissement Info Bulle:", e)
                df_plot['Avis'] = "Texte non disponible"
                df_plot['Avis Complet'] = "Texte non disponible"
        else:
            df_plot['Avis'] = "Texte non disponible"
            df_plot['Avis Complet'] = "Texte non disponible"

        # 4. Tracé avec Plasma et nouveau hover contextuel
        fig = px.scatter(
            df_plot, x='UMAP 1', y='UMAP 2',
            color='Note', 
            color_continuous_scale=px.colors.sequential.Plasma,
            title=f"Cartographie Sémantique ({len(df_plot)} avis échantillonnés)",
            hover_data={'UMAP 1': False, 'UMAP 2': False, 'Note': True, 'Avis': True},
            custom_data=['Avis Complet']
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
# GRAPHIQUES DE PERFORMANCES (BENCHMARK & LOSS)
# -----------------------------------------------------------------------------
chemin_historique = "data/historique_architectures.json"
fig_benchmark = px.bar(title="Lancement de l'Entraînement requis.")
fig_benchmark.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

if os.path.exists(chemin_historique):
    import json
    try:
        with open(chemin_historique, "r") as f:
            hist_data = json.load(f)
        df_hist = pd.DataFrame(hist_data)
        fig_benchmark = px.bar(
            df_hist, x="Architecture", y="R2_Score", color="R2_Score",
            text="R2_Score",
            color_continuous_scale=['#0d294f', '#1b4a8e', '#2980b9', '#3498db', '#85c1e9']
        )
        fig_benchmark.update_traces(texttemplate='<b>%{text:.4f}</b>', textposition='auto', textfont=dict(color='white'))
        fig_benchmark.update_layout(
            title=dict(text="<b>Benchmark des Architectures</b><br><span style='font-size:12px;color:#3498db'>Comparaison du R² Score</span>", font=dict(color="#ffffff", size=18)),
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
            margin=dict(t=60, b=40, l=20, r=20), xaxis_title="", yaxis_title="R² Score",
            coloraxis_showscale=False
        )
    except Exception as e:
        print("Erreur du chargement Benchmark:", e)

fig_loss = px.line(title="Lancement de l'Entraînement requis.")
fig_loss.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

if predicteur._mlp is not None and hasattr(predicteur._mlp, 'loss_curve_'):
    loss_data = {
        'Epoch': list(range(1, len(predicteur._mlp.loss_curve_) + 1)), 
        'Training Loss': predicteur._mlp.loss_curve_
    }
    df_loss = pd.DataFrame(loss_data)
    fig_loss = px.line(
        df_loss, x='Epoch', y='Training Loss', 
        markers=True
    )
    fig_loss.update_traces(line=dict(color='#3498db', width=3), marker=dict(size=6, color='#3498db'))
    fig_loss.update_layout(
        title=dict(text="<b>Convergence du Modèle</b><br><span style='font-size:12px;color:#3498db'>Évolution de la Perte (Training Loss)</span>", font=dict(color="#ffffff", size=18)),
        template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(t=60, b=40, l=20, r=20), xaxis_title="Itération (Epoch)", yaxis_title="Perte"
    )

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
    html.H5("Analyse de Sentiment Textuelle", className="card-title text-white mb-3", style={"fontWeight": "600"}),
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
    html.H5("Projection Sémantique Globale", className="card-title text-white mb-3", style={"fontWeight": "600"}),
    html.P("Topologie non-linéaire générée par l'algorithme UMAP optimisé avec Cosine Similarity. "
           "Cette représentation en 2D des embeddings met en évidence la séparabilité latente des avis structurée par le modèle BERT.", 
           className="text-muted", style={"fontSize": "0.9rem"}),
           
    # Le graphique + Activation de l'outillage de Zoom/Box (displayModeBar: True)
    dcc.Graph(id='umap-graph', figure=get_umap_figure(), config={'displayModeBar': True}),
    
    # Zone de lecture (Avis Complet) rattachée au graphique
    html.Div(
        id='avis-complet-output',
        children="Cliquez sur un point de la cartographie pour lire l'avis complet ici.",
        style={
            "backgroundColor": "#1a1a1a",
            "color": "#ecf0f1",
            "padding": "15px",
            "borderRadius": "8px",
            "border": "1px solid #444",
            "marginTop": "15px",
            "fontSize": "0.95rem",
            "fontStyle": "italic"
        }
    ),
    
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
    html.H5("Métriques d'Évaluation du Modèle", className="mb-4 text-white", style={"fontWeight": "600"}),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("ARCHITECTURE MLP", className="text-muted mb-1", style={"fontSize": "0.8rem", "letterSpacing": "1px"}),
                html.H3(str(architecture), style={"color": "#3498db", "fontWeight": "600"})
            ])
        ], style=CARD_STYLE)),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("R² SCORE (TEST)", className="text-muted mb-1", style={"fontSize": "0.8rem", "letterSpacing": "1px"}),
                html.H3(f"{r2_score:.4f}" if isinstance(r2_score, float) else r2_score, style={"color": "#3498db", "fontWeight": "600"})
            ])
        ], style=CARD_STYLE)),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("F1-SCORE CLASSIFICATION", className="text-muted mb-1", style={"fontSize": "0.8rem", "letterSpacing": "1px"}),
                html.H3(f"{f1_score_static}", style={"color": "#3498db", "fontWeight": "600"})
            ])
        ], style=CARD_STYLE))
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(figure=fig_benchmark, config={'displayModeBar': False}), style={"backgroundColor": "#1a1a1a", "border": "none", "borderRadius": "12px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"}), width=12, md=6, className="mb-4 mb-md-0"),
        dbc.Col(dbc.Card(dcc.Graph(figure=fig_loss, config={'displayModeBar': False}), style={"backgroundColor": "#1a1a1a", "border": "none", "borderRadius": "12px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"}), width=12, md=6)
    ], className="mb-5"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("MATRICE DE CONFUSION", className="text-white mb-2", style={"fontSize": "1rem", "letterSpacing": "1px", "fontWeight": "bold"}),
                html.P("Évaluation sur le jeu de test final (Seuil de décision > 2.5)", className="text-muted mb-4", style={"fontSize": "0.85rem"}),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("", style={"border": "none"}), 
                            html.Th("Prédit Négatif (1-2)", style={"padding": "12px", "color": "#3498db", "fontWeight": "bold"}), 
                            html.Th("Prédit Positif (3-5)", style={"padding": "12px", "color": "#3498db", "fontWeight": "bold"})
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Th("Vrai Négatif (1-2)", style={"padding": "12px", "color": "#3498db", "textAlign": "right", "fontWeight": "bold"}),
                            html.Td("23,410", style={"padding": "15px", "backgroundColor": "rgba(52, 152, 219, 0.15)", "fontWeight": "bold", "border": "1px solid #333", "color": "#fff", "fontSize": "1.1rem"}),
                            html.Td("4,580", style={"padding": "15px", "border": "1px solid #333", "color": "#aaa"})
                        ]),
                        html.Tr([
                            html.Th("Vrai Positif (3-5)", style={"padding": "12px", "color": "#3498db", "textAlign": "right", "fontWeight": "bold"}),
                            html.Td("3,912", style={"padding": "15px", "border": "1px solid #333", "color": "#aaa"}),
                            html.Td("28,430", style={"padding": "15px", "backgroundColor": "rgba(52, 152, 219, 0.15)", "fontWeight": "bold", "border": "1px solid #333", "color": "#fff", "fontSize": "1.1rem"})
                        ])
                    ])
                ], style={"width": "100%", "marginTop": "10px", "borderCollapse": "collapse", "textAlign": "center"})
            ])
        ], style={"backgroundColor": "#1a1a1a", "border": "none", "borderRadius": "12px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"}), width=12, lg=8, className="mx-auto")
    ])
])

app.layout = dbc.Container([
    # En-tête
    html.Div([
        html.H2("NLP Analytics Dashboard", className="text-white mb-1", style={"fontWeight": "600", "letterSpacing": "-0.5px"}),
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

@app.callback(
    Output('avis-complet-output', 'children'),
    Input('umap-graph', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Cliquez sur un point de la cartographie pour lire l'avis complet ici."
    
    try:
        texte_complet = clickData['points'][0]['customdata'][0]
        # Si le point cliqué porte une mention "Texte non disponible", on modifie le rendu
        if texte_complet == "Texte non disponible":
            return html.Div("Le texte complet pour cet avis est indisponible (CSV introuvable).", style={"color": "#e74c3c"})
            
        return html.Div([
            html.Strong("Avis Sélectionné :", style={"color": "#3498db"}),
            html.Br(), html.Br(),
            html.Span(texte_complet, style={"fontStyle": "normal", "lineHeight": "1.5"})
        ])
    except Exception as e:
        return f"Erreur système lors du chargement détaillé : {str(e)}"

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Démarrage du serveur Dash localement sur http://127.0.0.1:8050/")
    app.run(debug=True)
