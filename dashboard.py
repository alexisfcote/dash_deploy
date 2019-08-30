import plotly.express as px
import plotly.graph_objects as go
import flask
import dash
import dash_auth
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import scipy
import scipy.signal

import pandas as pd
import numpy as np
from pathlib import Path


def rgb2rgba(color, alpha):
    col = color.replace('rgb', 'rgba').replace(')', ',{})'.format(alpha))
    return col


FREQ = 100
df = pd.read_pickle('data.pkl')
df.loc[~df['Étudiant - Rate Valid'], 'Étudiant - Rate'] = np.nan
df.loc[~df['Étudiant - EDA valid'], "Étudiant - EDA phasic"] = np.nan

col_options = [dict(label=x, value=x) for x in df.columns]
dyade_options = [dict(label=str(x), value=x)
                 for x in sorted(df.dyade.unique())]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

VALID_USERNAME_PASSWORD_PAIRS = {
    'Éduqués': 'stressés'
}

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


app.layout = html.Div(
    [
        html.H1("Signaux physio Eduquées Stressées"),
        html.Div(
            [
                html.Div(
                    [html.H3(["Dyade " + ":", ]),
                     html.P(
                         [dcc.Dropdown(id="dyade", options=dyade_options, value=1)]),
                     html.P(["Nombre de sommets" + ":", ]),
                     html.P(
                         [dcc.Slider(id="num_peaks",
                                        min=0,
                                        max=15,
                                        step=1,
                                        value=5)]),
                     html.P(["", dcc.Input(id="num_peaks_disp", readOnly=True)]),
                     ],
                    style={'padding': 20}),
                html.P(["HR mean " + ":", dcc.Input(id="hr_moy", readOnly=True)]),
                html.P(["HR std " + ":", dcc.Input(id="hr_std", readOnly=True)]),
            ],
            style={"width": "15%", "float": "left"},
        ),
        html.Div(
            [
                html.P(["Point Event " + ":",
                        dcc.Textarea(id="point_count", readOnly=True), 
                        "State Event " + ":",
                        dcc.Textarea(id="state", readOnly=True)]),
            ],
        ),
        dcc.Graph(id="graph", style={
                  "width": "85%", "display": "inline-block"}),

        html.Div(
            [
                html.P(["EDA mean " + ":", dcc.Input(id="eda_moy", readOnly=True)]),
                html.P(["EDA std " + ":", dcc.Input(id="eda_std", readOnly=True)]),
            ],
            style={"width": "15%", "float": "left"},
        ),
        dcc.Graph(id="graph2", style={
                  "width": "85%", "display": "inline-block"}),
    ]
)


@app.callback(
    Output("num_peaks_disp", "value"),
    [
    Input("num_peaks", "value"),
])
def update_num_peaks_ind(num_peaks):
    return int(num_peaks)


@app.callback([
    Output("graph", "figure"),
    Output("hr_moy", "value"),
    Output("hr_std", "value"),
],
    [
    Input("dyade", "value"),
    Input("num_peaks", "value"),
])
def update_graph(dyade, num_peaks):
    df_dyade = df[df.dyade == dyade]
    df_dyade = df_dyade.iloc[::10]
    rate = df_dyade['Étudiant - Rate']
    fig = make_figure(dyade, df_dyade, 'Étudiant - Rate',
                      "Étudiant HR", num_peaks)
    return fig, f'{rate.mean():.02f}', f'{rate.std():.03f}'


@app.callback([
    Output("graph2", "figure"),
    Output("eda_moy", "value"),
    Output("eda_std", "value"),
],
    [
    Input("dyade", "value"),
    Input("num_peaks", "value"),
])
def update_graph2(dyade, num_peaks):
    df_dyade = df[df.dyade == dyade]
    df_dyade = df_dyade.iloc[::10]
    rate = df_dyade['Étudiant - EDA phasic']
    fig = make_figure(dyade, df_dyade, 'Étudiant - EDA phasic',
                      "Étudiant - EDA phasic", num_peaks)
    return fig, f'{rate.mean():.02f}', f'{rate.std():.03f}'


def make_figure(dyade, df_dyade, col, title, num_peaks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_dyade['time'], y=df_dyade[col], mode='lines', line=dict(
        color='black', width=1), name=title))

    if 'phasic' in title:
        envelop = df_dyade['Étudiant - EDA phasic envelop']
    else:
        envelop = df_dyade[col].rolling(8*FREQ//10, center=True).mean()

    fig.add_trace(go.Scatter(x=df_dyade['time'], y=envelop, mode='lines', line=dict(
        color='brown', width=1), name=title + ' enveloppe'))
    
    envelop[np.isnan(envelop)] = 0
    peak_ind = scipy.signal.find_peaks(envelop, distance=FREQ//10*8)[0]
    peak_value = envelop.values[peak_ind]

    sorted_peaks = sorted(zip(peak_ind, peak_value),
                          key=lambda x: x[1], reverse=True)
    peak_ind, peak_value = map(list, zip(*sorted_peaks))

    peak_ind = peak_ind[:int(num_peaks)]
    fig.add_trace(go.Scatter(x=df_dyade['time'].values[peak_ind], y=envelop.values[peak_ind], mode='markers', marker={
                  'symbol': 'x', 'size': 15}, name='Sommets'))

    #fig = px.line(df_dyade, x='time', y=col, )
    rate = df_dyade[col]

    def plot_cat(cat, color):
        fig.add_trace(go.Scatter(x=df_dyade.time, y=(df_dyade.Behavior_clean == cat).astype(float) * np.max(rate),
                                 fill=None,
                                 mode='lines',
                                 line=dict(width=0),
                                 showlegend=False,
                                 hoverinfo='skip',
                                 ))
        fig.add_trace(go.Scatter(
            x=df_dyade.time,
            y=(df_dyade.Behavior_clean == cat).astype(float) * np.min(rate),
            fill='tonexty',  # fill area between trace0 and trace1
            fillcolor=rgb2rgba(color, 0.3),
            mode='lines',
            line=dict(width=0),
            showlegend=True,
            name=cat,
            hoverinfo='skip',
        ))

    plot_cat('collaboration', px.colors.qualitative.Set1[3])
    plot_cat('ambigu', px.colors.qualitative.Set1[5])
    plot_cat('retrait', px.colors.qualitative.Set1[7])

    def plot_point(cat, color=None):
        def flatten(l): return [item for sublist in l for item in sublist]
        vals = df_dyade.time[df_dyade.Behavior_point_clean == cat].values
        x = flatten([[val, val, np.nan] for val in vals])
        y = flatten([[np.min(rate), np.max(rate), np.nan] for _ in vals])

        fig.add_scatter(mode='lines', x=x, y=y, name=cat,
                        hovertemplate="time: %{x}", line=dict(color=color, width=1))

    plot_point('critique', 'red')
    plot_point('ignore', 'blue')
    plot_point('soutien', 'green')

    fig.update_layout(
        title=go.layout.Title(
            text=title,
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=title,
            )
        ),
    )
    fig.update_yaxes(range=[np.min(rate), np.max(rate)])

    return fig


@app.callback(
    [
        Output("point_count", "value"),
        Output("state", "value"),
    ],
    [
    Input("dyade", "value"),
])
def update_stats(dyade):
    df_dyade = df[df.dyade == dyade]
    values = df_dyade.Behavior_point_clean.value_counts().to_dict()
    del values['']
    
    states = ((df_dyade.Behavior_clean.value_counts()/len(df_dyade)*100).astype(int).astype(str) + '%').to_dict()
    del states['']

    return (str(values)[1:-1], str(states)[1:-1])


if __name__ == "__main__":
    app.run_server(debug=True)
