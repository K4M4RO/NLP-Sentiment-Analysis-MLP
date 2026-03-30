import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os
import json
import joblib

# Import de notre IA
from composants.ia_notes import NotesPredicteur

# -----------------------------------------------------------------------------
# INITIALISATION DU MODÈLE
# -----------------------------------------------------------------------------
print("Initialisation du modèle pour l'application Dash (Design Refondu)...")
predicteur = NotesPredicteur()
predicteur.charger_cerveau() # Charge le meilleur modèle sélectionné (Early Stopping)

# --- CHARGEMENT DE LA CONFIGURATION DES GENRES (pour le Dropdown) ---
chemin_genres_config = "data/genres_config.json"
options_genres = []
if os.path.exists(chemin_genres_config):
    with open(chemin_genres_config, "r", encoding="utf-8") as f:
        genres_config = json.load(f)
    # On retire le préfixe "genre_" pour l'affichage humain
    options_genres = [
        {"label": col.replace("genre_", ""), "value": col.replace("genre_", "")}
        for col in genres_config.get("colonnes_genre", [])
        if col != "genre_Inconnu"  # On masque "Inconnu" (c'est le fallback automatique)
    ]
    print(f"Genres chargés pour le Dropdown : {len(options_genres)} options.")
else:
    print("Aucun fichier genres_config.json trouvé. Dropdown désactivé.")

# Variables pour Performance
r2_score = "N/A"
architecture = "N/A"
f1_val = "N/A"
mat_conf = "N/A"
class_report = "N/A"

try:
    if hasattr(predicteur, "_score_ia") and predicteur._score_ia:
        r2_score = predicteur._score_ia.get('R2', "N/A")
        f1_val = predicteur._score_ia.get('f1_score', "N/A")
        if f1_val != "N/A":
            f1_val = round(float(f1_val), 3)
        mat_conf = predicteur._score_ia.get('confusion_matrix', "N/A")
        class_report = predicteur._score_ia.get('classification_report', "N/A")
        
    if hasattr(predicteur, "_mlp") and predicteur._mlp is not None:
        architecture = predicteur._mlp.hidden_layer_sizes
        
    # Fallback vers l'historique json si manquant
    if f1_val == "N/A" or mat_conf == "N/A" or class_report == "N/A":
        chemin_historique = "data/historique_architectures.json"
        if os.path.exists(chemin_historique):
            with open(chemin_historique, "r") as f:
                hist_data = json.load(f)
            if hist_data:
                dernier = hist_data[-1]
                if f1_val == "N/A":
                    fval = dernier.get("f1_score", "N/A")
                    if fval != "N/A":
                        f1_val = round(float(fval), 3)
                mat_conf = mconf if (mconf := dernier.get("confusion_matrix")) else mat_conf
                class_report = crep if (crep := dernier.get("classification_report")) else class_report
                r2_score = r2_score if r2_score != "N/A" else dernier.get("R2_Score", "N/A")

    # Fallback ultime : Calcul depuis le DataFrame UMAP si disponible
    if (f1_val == "N/A" or mat_conf == "N/A" or class_report == "N/A") and df_umap_global is not None:
        if 'Note_Vraie' in df_umap_global.columns and 'Note_Predite' in df_umap_global.columns:
            from sklearn.metrics import confusion_matrix, f1_score, classification_report
            y_true_bin = (df_umap_global['Note_Vraie'] > 2.5).astype(int)
            y_pred_bin = (df_umap_global['Note_Predite'] > 2.5).astype(int)
            if mat_conf == "N/A":
                mat_conf = confusion_matrix(y_true_bin, y_pred_bin).tolist()
            if f1_val == "N/A":
                f1_val = round(f1_score(y_true_bin, y_pred_bin), 3)
            if class_report == "N/A":
                class_report = classification_report(y_true_bin, y_pred_bin, output_dict=True, zero_division=0)
except Exception as e:
    print(f"Erreur chargement perf: {e}")

# Préparation HTML Matrice de confusion
tn, fp, fn, tp = "N/A", "N/A", "N/A", "N/A"
if isinstance(mat_conf, list) and len(mat_conf) == 2:
    tn, fp = mat_conf[0]
    fn, tp = mat_conf[1]
    tn, fp, fn, tp = f"{tn:,}".replace(',', ' '), f"{fp:,}".replace(',', ' '), f"{fn:,}".replace(',', ' '), f"{tp:,}".replace(',', ' ')

# Préparation HTML Rapport Classification
report_rows = []
if class_report != "N/A" and isinstance(class_report, dict):
    labels_map = {
        "0": "Inintéressant (<=2.5)",
        "1": "Intéressant (>2.5)",
        "macro avg": "Macro Avg",
        "weighted avg": "Weighted Avg"
    }
    for key in ["0", "1", "macro avg", "weighted avg"]:
        if key in class_report:
            row_data = class_report[key]
            bg_color = "rgba(255, 255, 255, 0.03)" if "avg" in key else "transparent"
            font_weight = "bold" if "avg" in key else "normal"
            support = int(row_data['support']) if 'support' in row_data else ""
            
            report_rows.append(
                html.Tr([
                    html.Td(labels_map.get(key, key), style={"padding": "12px", "color": "#3498db" if "avg" not in key else "#aaa", "fontWeight": "bold", "borderBottom": "1px solid #333", "textAlign": "left"}),
                    html.Td(f"{row_data['precision']:.3f}", style={"padding": "12px", "color": "#fff", "borderBottom": "1px solid #333", "textAlign": "right", "fontWeight": font_weight}),
                    html.Td(f"{row_data['recall']:.3f}", style={"padding": "12px", "color": "#fff", "borderBottom": "1px solid #333", "textAlign": "right", "fontWeight": font_weight}),
                    html.Td(f"{row_data.get('f1-score', 0):.3f}", style={"padding": "12px", "color": "#3498db" if "avg" not in key else "#fff", "fontWeight": "bold" if "avg" not in key else "normal", "borderBottom": "1px solid #333", "textAlign": "right", "backgroundColor": "rgba(52, 152, 219, 0.1)" if "avg" not in key else "transparent"}),
                    html.Td(f"{support:,}".replace(',', ' ') if support else "", style={"padding": "12px", "color": "#aaa", "borderBottom": "1px solid #333", "textAlign": "right"})
                ], style={"backgroundColor": bg_color})
            )

class_report_component = html.Div([
    html.Table([
        html.Thead(
            html.Tr([
                html.Th("Classe", style={"padding": "12px", "color": "#888", "borderBottom": "2px solid #555", "textAlign": "left"}), 
                html.Th("Précision", style={"padding": "12px", "color": "#888", "borderBottom": "2px solid #555", "textAlign": "right"}), 
                html.Th("Rappel", style={"padding": "12px", "color": "#888", "borderBottom": "2px solid #555", "textAlign": "right"}),
                html.Th("F1-Score", style={"padding": "12px", "color": "#888", "borderBottom": "2px solid #555", "textAlign": "right"}),
                html.Th("Support", style={"padding": "12px", "color": "#888", "borderBottom": "2px solid #555", "textAlign": "right"}),
            ])
        ),
        html.Tbody(report_rows)
    ], style={"width": "100%", "marginTop": "10px", "borderCollapse": "collapse"})
]) if report_rows else html.Div("Rapport non disponible.", style={"color": "#aaa", "fontStyle": "italic", "padding": "20px"})

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

DROPDOWN_STYLE = {
    "backgroundColor": "#1a1a1a",
    "color": "#fff",
    "border": "1px solid #444",
    "borderRadius": "8px",
    "marginBottom": "12px"
}

# -----------------------------------------------------------------------------
# PRÉPARATION DES DONNÉES UMAP (Global)
# -----------------------------------------------------------------------------
chemin_csv_umap = "data/donnees_dashboard_umap.csv"
df_umap_global = None

if os.path.exists(chemin_csv_umap):
    df_umap_global = pd.read_csv(chemin_csv_umap)
    
    # Textes pour le survol et le clic
    df_umap_global['Avis'] = df_umap_global['Texte'].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x))
    df_umap_global['Avis Complet'] = df_umap_global['Texte']
    
    print(f"✅ Données UMAP Dashboard chargées : {len(df_umap_global)} points.")
else:
    print(f"Avertissement : Fichier {chemin_csv_umap} introuvable. Effectuez '--project'.")

def get_umap_figure(range_notes=[1, 5], genres_selection=None):
    if df_umap_global is not None:
        # Filtrage dynamique selon le slider sur la note vraie
        mask = (df_umap_global['Note_Vraie'] >= range_notes[0]) & (df_umap_global['Note_Vraie'] <= range_notes[1])
        
        # Filtrage dynamique selon les genres (si liste non vide)
        if genres_selection:
            mask = mask & df_umap_global['Genre'].isin(genres_selection)
            
        df_plot = df_umap_global[mask]

        has_genre = 'Genre' in df_plot.columns and df_plot['Genre'].nunique() > 1
        
        # Le graphique utilise UMAP_X, UMAP_Y et est coloré par la Note Prédite par le modèle
        fig = px.scatter(
            df_plot, x='UMAP_X', y='UMAP_Y',
            color='Note_Predite', 
            color_continuous_scale=px.colors.sequential.Plasma,
            symbol='Genre' if has_genre else None,
            title=f"Cartographie Sémantique IA ({len(df_plot)} avis)",
            hover_data={
                'UMAP_X': False, 
                'UMAP_Y': False, 
                'Note_Predite': ':.2f', 
                'Note_Vraie': True, 
                'Avis': True, 
                'Genre': True
            } if has_genre else {
                'UMAP_X': False, 
                'UMAP_Y': False, 
                'Note_Predite': ':.2f', 
                'Note_Vraie': True, 
                'Avis': True
            },
            custom_data=['Avis Complet']
        )
        
        # Optimisation visuelle brillante (taille: 5, opacité: 0.8)
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        
        # NETTOYAGE DU GRAPHIQUE
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
                    text="Note Prédite",
                    side="top",
                    font=dict(color="#eee", size=14)
                ),
                thickness=12,
                len=0.8,
                yanchor="middle",
                y=0.5,
                tickfont=dict(color="#aaa")
            ),
            legend=dict(
                orientation="h", 
                y=-0.15, 
                x=0.5, 
                xanchor="center", 
                bgcolor='rgba(0,0,0,0)', 
                font=dict(color="#eee", size=12)
            )
        )
        return fig
    else:
        fig = px.scatter(title="Veuillez générer l'UMAP via 'python main.py --project'")
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
    epochs = list(range(1, len(predicteur._mlp.loss_curve_) + 1))
    loss_data = {
        'Epoch': epochs, 
        'Entraînement': predicteur._mlp.loss_curve_
    }
    if hasattr(predicteur._mlp, 'val_loss_curve_'):
        loss_data['Validation'] = predicteur._mlp.val_loss_curve_
        
    df_loss = pd.DataFrame(loss_data)
    y_cols = ['Entraînement', 'Validation'] if 'Validation' in df_loss.columns else ['Entraînement']
    
    fig_loss = px.line(
        df_loss, x='Epoch', y=y_cols, 
        markers=True
    )
    
    color_map = {'Entraînement': '#3498db', 'Validation': '#e74c3c'}
    fig_loss.for_each_trace(lambda trace: trace.update(
        line=dict(color=color_map[trace.name], width=3), 
        marker=dict(size=6, color=color_map[trace.name])
    ))
    
    fig_loss.update_layout(
        title=dict(text="<b>Convergence du Modèle</b><br><span style='font-size:12px;color:#3498db'>Évolution de la Perte (Training vs Validation)</span>", font=dict(color="#ffffff", size=18)),
        template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(t=60, b=40, l=20, r=20), xaxis_title="Itération (Epoch)", yaxis_title="Perte",
        legend_title_text='',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
            /* Dark Mode Dropdown */
            .Select-control { background-color: #1a1a1a !important; border-color: #444 !important; color: #fff !important; }
            .Select-menu-outer { background-color: #1a1a1a !important; border-color: #444 !important; }
            .Select-option { background-color: #1a1a1a !important; color: #ccc !important; }
            .Select-option.is-focused { background-color: #333 !important; color: #fff !important; }
            .Select-value-label { color: #fff !important; }
            .Select-placeholder { color: #888 !important; }
            .Select-input input { color: #fff !important; }
            .VirtualizedSelectOption { background-color: #1a1a1a !important; color: #ccc !important; }
            .VirtualizedSelectFocusedOption { background-color: #333 !important; color: #fff !important; }
            
            /* Dark Mode Dropdown Multi-value */
            .Select-value { background-color: #3a3f48 !important; border: 1px solid #555 !important; color: #fff !important; }
            .Select-value-icon { border-right: 1px solid #555 !important; color: #fff !important; }
            .Select-value-icon:hover { background-color: #e74c3c !important; color: #fff !important; }
            
            /* RangeSlider Accentuation */
            .rc-slider-track { background-color: #9b59b6 !important; }
            .rc-slider-handle { border: solid 2px #9b59b6 !important; background-color: #222 !important; }
            .rc-slider-dot-active { border-color: #9b59b6 !important; }
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
    
    # --- DROPDOWN GENRE (chargé depuis genres_config.json) ---
    html.Label("Genre du livre :", style={"color": "#aaa", "fontSize": "0.85rem", "marginBottom": "6px", "display": "block"}),
    dcc.Dropdown(
        id="dropdown-genre",
        options=options_genres,
        value=None,
        placeholder="Sélectionner un genre (optionnel — défaut : Inconnu)",
        searchable=True,
        clearable=True,
        style=DROPDOWN_STYLE
    ),
    
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
           "Cette représentation 2D projette les activations de la dernière couche cachée du réseau de neurones (MLP)."
           "Elle illustre comment le modèle réorganise les embeddings sémantiques de BERT pour tracer sa frontière de décision et séparer les sentiments.", 
           className="text-muted", style={"fontSize": "0.9rem"}),
           
    # --- PANNEAU DE CONTRÔLE (FLEXBOX) ---
    html.Div([
        # Filtre Slider des Notes
        html.Div([
            html.H6("Filtre des Notes (Vérité Terrain) :", style={'color': '#E2E2E2', 'fontWeight': 'bold', 'marginBottom': '10px', 'fontFamily': 'sans-serif', 'fontSize': '0.9rem'}),
            dcc.RangeSlider(
                id='slider-note-umap',
                min=1,
                max=5,
                step=0.5,
                value=[1, 5],
                marks={i: {'label': str(i), 'style': {'color': '#aaa', 'fontWeight': 'bold'}} for i in range(1, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={"flex": "1", "minWidth": "300px", "paddingRight": "20px"}),
        
        # Filtre Dropdown des Genres
        html.Div([
            html.H6("Filtre des Genres :", style={'color': '#E2E2E2', 'fontWeight': 'bold', 'marginBottom': '10px', 'fontFamily': 'sans-serif', 'fontSize': '0.9rem'}),
            dcc.Dropdown(
                id="dropdown-genre-umap",
                options=options_genres,
                value=[],
                multi=True,
                placeholder="Sélectionnez un ou plusieurs genres (vide = tous)",
                style={"backgroundColor": "#22252b", "color": "#fff", "border": "1px solid #3a3f48", "borderRadius": "8px"}
            )
        ], style={"flex": "1", "minWidth": "300px"})
    ], style={
        "display": "flex", 
        "justifyContent": "space-between", 
        "gap": "20px", 
        "padding": "20px", 
        "backgroundColor": "#1a1a1a", 
        "borderRadius": "8px", 
        "border": "1px solid #333",
        "marginBottom": "20px",
        "flexWrap": "wrap"
    }),

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
    )
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
                html.H3(f"{f1_val}", style={"color": "#3498db", "fontWeight": "600"})
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
                html.P("Évaluation dynamique (Seuil > 2.5)", className="text-muted mb-4", style={"fontSize": "0.85rem"}),
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
                            html.Td(f"{tn}", style={"padding": "15px", "backgroundColor": "rgba(52, 152, 219, 0.15)", "fontWeight": "bold", "border": "1px solid #333", "color": "#fff", "fontSize": "1.1rem"}),
                            html.Td(f"{fp}", style={"padding": "15px", "border": "1px solid #333", "color": "#aaa"})
                        ]),
                        html.Tr([
                            html.Th("Vrai Positif (3-5)", style={"padding": "12px", "color": "#3498db", "textAlign": "right", "fontWeight": "bold"}),
                            html.Td(f"{fn}", style={"padding": "15px", "border": "1px solid #333", "color": "#aaa"}),
                            html.Td(f"{tp}", style={"padding": "15px", "backgroundColor": "rgba(52, 152, 219, 0.15)", "fontWeight": "bold", "border": "1px solid #333", "color": "#fff", "fontSize": "1.1rem"})
                        ])
                    ])
                ], style={"width": "100%", "marginTop": "10px", "borderCollapse": "collapse", "textAlign": "center"})
            ])
        ], style={"backgroundColor": "#1a1a1a", "border": "none", "borderRadius": "12px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "height": "100%"}), width=12, lg=5, className="mb-4 mb-lg-0"),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("RAPPORT DE CLASSIFICATION", className="text-white mb-2", style={"fontSize": "1rem", "letterSpacing": "1px", "fontWeight": "bold"}),
                html.P("Métriques détaillées pour chaque classe", className="text-muted mb-4", style={"fontSize": "0.85rem"}),
                class_report_component
            ])
        ], style={"backgroundColor": "#1a1a1a", "border": "none", "borderRadius": "12px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "height": "100%"}), width=12, lg=7)
    ], className="mb-5")
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
    State("input-texte", "value"),
    State("dropdown-genre", "value")
)
def update_prediction(n_clicks, texte, genre_selectionne):
    if not n_clicks or not texte:
        return dash.no_update
    
    try:
        # Passe le genre sélectionné (ou None → backend activera genre_Inconnu)
        score = predicteur.predire_score(texte, genre=genre_selectionne)
        is_interessant = score > 2.5
        texte_resultat = "Intéressant" if is_interessant else "Inintéressant"
        genre_affiche = genre_selectionne if genre_selectionne else "Inconnu"
        
        couleur_bord = "#2ecc71" if is_interessant else "#e74c3c"
        couleur_texte = "#2ecc71" if is_interessant else "#e74c3c"
        
        return dbc.Card([
            dbc.CardBody([
                html.H6("RÉSULTAT DE L'INFÉRENCE", className="text-muted mb-2", style={"fontSize": "0.75rem", "letterSpacing": "1px"}),
                html.Div([
                    html.Div([
                        html.Span("Genre : ", style={"color": "#ccc"}),
                        html.Strong(genre_affiche, style={"color": "#3498db", "fontSize": "1rem", "marginLeft": "10px"})
                    ], className="mb-2"),
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
    Input("slider-note-umap", "value"),
    Input("dropdown-genre-umap", "value")
)
def update_umap_graph(range_val, genres_val):
    if range_val is None:
        return dash.no_update
    return get_umap_figure(range_val, genres_val)

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
