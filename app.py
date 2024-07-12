import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback
from dash import dash_table
from data_processing import data_processing 


key_metrics, sales_metrics, listings_metrics, mints_metrics, mint_to_sales_fig, listing_to_sales_fig, daily_sales_fig, daily_vol_fig, highest_selling_domains_fig, monthly_box_sales_metrics, latest_box_domains_sales, highest_selling_domains, listings_growth_rate_fig, historical_listing_to_sales, latest_box_listings, daily_mint_metrics_fig, latest_box_domains_mints, model_prep, value_domain = data_processing()


external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css', 
    '/assets/styles.css'
]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={'backgroundColor': 'var(--color-background)'}, children=[
    html.H1(
        children='.box Domains Dashboard',
        style={
            'textAlign': 'center',
            'color': 'var(--wcm-color-fg-1)',
            'fontSize': '36px',
            'fontWeight': 'bold',
            'marginBottom': '20px'
        }
    ),
    html.Br(),
    html.H2('.box Domain Valuator', style={
        'color': 'var(--wcm-color-fg-2)', 
        'textAlign': 'center', 
        'marginBottom': '20px'
    }),
    html.Div([
        html.Label("Input Domain Name:", style={
            'color': 'var(--wcm-color-fg-1)', 
            'marginRight': '10px',
            'fontWeight': 'bold'
        }),
        dcc.Input(
            id='valuator-input',
            value='example',
            type='text',
            style={
                'padding': '10px',
                'borderRadius': 'var(--wcm-input-border-radius)',
                'border': '1px solid var(--color-border)',
                'marginRight': '10px'
            },
            pattern='[^.]*'  # Regex pattern to disallow '.' character
        ),
        html.Button('Submit', id='submit-button', n_clicks=0, style={
            'padding': '10px 20px', 
            'borderRadius': 'var(--wcm-button-border-radius)', 
            'backgroundColor': 'var(--wcm-accent-color)', 
            'color': 'var(--wcm-accent-fill-color)',
            'border': 'none',
            'cursor': 'pointer'
        })
    ], style={
        'display': 'flex', 
        'alignItems': 'center', 
        'justifyContent': 'center', 
        'marginBottom': '20px'
    }),
    html.Br(),
    html.Div(id='valuator-output', style={
        'color': 'var(--wcm-color-fg-1)', 
        'textAlign': 'center', 
        'marginTop': '20px'
    }),
    html.Br(),
    html.H2('Key Metrics', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(className='metrics-container', children=[
        html.Div(className='metric', children=[
            html.Span(metric["label"], className='label'),
            html.Span(f"{metric['value']}{metric['unit']}", className='value')
        ]) for metric in key_metrics
    ]),
    html.Br(),
    dcc.Graph(id='mint to sales', figure=mint_to_sales_fig),
    dcc.Graph(id='listings to sales', figure=listing_to_sales_fig),
    
    html.H2('Sales', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(className='metrics-container', children=[
        html.Div(className='metric', children=[
            html.Span(metric["label"], className='label'),
            html.Span(f"{metric['value']}{metric['unit']}", className='value')
        ]) for metric in sales_metrics
    ]),
    html.Br(),
    dcc.Graph(id='daily_sales_count', figure=daily_sales_fig),
    dcc.Graph(id='daily_sales_vol', figure=daily_vol_fig),
    dcc.Graph(id='highest selling', figure=highest_selling_domains_fig),
    html.Br(),
    html.H3('Monthly Sales Metrics', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(style={'display': 'flex', 'justify-content': 'center', 'padding': '10px'}, children=[
        html.Div(style={'width': '80%', 'max-width': '1000px'}, children=[
            dash_table.DataTable(
                id='monthly_sales_metrics',
                columns=[{"name": i, "id": i} for i in monthly_box_sales_metrics.columns],
                data=monthly_box_sales_metrics.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'var(--wcm-color-bg-2)',
                    'fontWeight': 'bold',
                    'color': 'var(--wcm-color-fg-1)'
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'font-family': 'var(--font-primary, "Inter")',
                    'backgroundColor': 'var(--color-background)',
                    'color': 'var(--wcm-color-fg-2)',
                    'padding': '10px',
                    'border': '1px solid var(--color-border)'
                },
                style_data={
                    'border': '1px solid var(--color-border)',
                    'padding': '10px',
                }
            )
        ])
    ]),
    html.Br(),
    html.H3('10 Latest Sales', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(style={'display': 'flex', 'justify-content': 'center', 'padding': '10px'}, children=[
        html.Div(style={'width': '80%', 'max-width': '1000px'}, children=[
            dash_table.DataTable(
                id='latest_sales',
                columns=[{"name": i, "id": i} for i in latest_box_domains_sales.columns],
                data=latest_box_domains_sales.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'var(--wcm-color-bg-2)',
                    'fontWeight': 'bold',
                    'color': 'var(--wcm-color-fg-1)'
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'font-family': 'var(--font-primary, "Inter")',
                    'backgroundColor': 'var(--color-background)',
                    'color': 'var(--wcm-color-fg-2)',
                    'padding': '10px',
                    'border': '1px solid var(--color-border)'
                },
                style_data={
                    'border': '1px solid var(--color-border)',
                    'padding': '10px',
                }
            )
        ])
    ]),
    html.Br(),
    html.H3('10 Highest Selling Domains', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(style={'display': 'flex', 'justify-content': 'center', 'padding': '10px'}, children=[
        html.Div(style={'width': '80%', 'max-width': '1000px'}, children=[
            dash_table.DataTable(
                id='highest_sales',
                columns=[{"name": i, "id": i} for i in highest_selling_domains.columns],
                data=highest_selling_domains.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'var(--wcm-color-bg-2)',
                    'fontWeight': 'bold',
                    'color': 'var(--wcm-color-fg-1)'
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'font-family': 'var(--font-primary, "Inter")',
                    'backgroundColor': 'var(--color-background)',
                    'color': 'var(--wcm-color-fg-2)',
                    'padding': '10px',
                    'border': '1px solid var(--color-border)'
                },
                style_data={
                    'border': '1px solid var(--color-border)',
                    'padding': '10px',
                }
            )
        ])
    ]),
    html.Br(),
    html.H2('Listings', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(className='metrics-container', children=[
        html.Div(className='metric', children=[
            html.Span(metric["label"], className='label'),
            html.Span(f"{metric['value']}{metric['unit']}", className='value')
        ]) for metric in listings_metrics
    ]),
    html.Br(),
    dcc.Graph(id='monthly listings growth', figure=listings_growth_rate_fig),
    html.H3('Historical Listings to Sales', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(style={'display': 'flex', 'justify-content': 'center', 'padding': '10px'}, children=[
        html.Div(style={'width': '80%', 'max-width': '1000px'}, children=[
            dash_table.DataTable(
                id='listings_to_sales',
                columns=[{"name": i, "id": i} for i in historical_listing_to_sales.columns],
                data=historical_listing_to_sales.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'var(--wcm-color-bg-2)',
                    'fontWeight': 'bold',
                    'color': 'var(--wcm-color-fg-1)'
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'font-family': 'var(--font-primary, "Inter")',
                    'backgroundColor': 'var(--color-background)',
                    'color': 'var(--wcm-color-fg-2)',
                    'padding': '10px',
                    'border': '1px solid var(--color-border)'
                },
                style_data={
                    'border': '1px solid var(--color-border)',
                    'padding': '10px',
                }
            )
        ])
    ]),
    html.H3('10 Latest Listings', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(style={'display': 'flex', 'justify-content': 'center', 'padding': '10px'}, children=[
        html.Div(style={'width': '80%', 'max-width': '1000px'}, children=[
            dash_table.DataTable(
                id='latest_listings',
                columns=[{"name": i, "id": i} for i in latest_box_listings.columns],
                data=latest_box_listings.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'var(--wcm-color-bg-2)',
                    'fontWeight': 'bold',
                    'color': 'var(--wcm-color-fg-1)'
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'font-family': 'var(--font-primary, "Inter")',
                    'backgroundColor': 'var(--color-background)',
                    'color': 'var(--wcm-color-fg-2)',
                    'padding': '10px',
                    'border': '1px solid var(--color-border)'
                },
                style_data={
                    'border': '1px solid var(--color-border)',
                    'padding': '10px',
                }
            )
        ])
    ]),

    html.H2('Mints', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(className='metrics-container', children=[
        html.Div(className='metric', children=[
            html.Span(metric["label"], className='label'),
            html.Span(f"{metric['value']}{metric['unit']}", className='value')
        ]) for metric in mints_metrics
    ]),
    html.Br(),
    dcc.Graph(id='daily_mints', figure=daily_mint_metrics_fig),
    html.Br(),
    html.H3('10 Latest Mints', style={'color': 'var(--wcm-color-fg-1)'}),
    html.Div(style={'display': 'flex', 'justify-content': 'center', 'padding': '10px'}, children=[
        html.Div(style={'width': '80%', 'max-width': '1000px'}, children=[
            dash_table.DataTable(
                id='latest_mints',
                columns=[{"name": i, "id": i} for i in latest_box_domains_mints.columns],
                data=latest_box_domains_mints.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'var(--wcm-color-bg-2)',
                    'fontWeight': 'bold',
                    'color': 'var(--wcm-color-fg-1)'
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'font-family': 'var(--font-primary, "Inter")',
                    'backgroundColor': 'var(--color-background)',
                    'color': 'var(--wcm-color-fg-2)',
                    'padding': '10px',
                    'border': '1px solid var(--color-border)'
                },
                style_data={
                    'border': '1px solid var(--color-border)',
                    'padding': '10px',
                }
            )
        ])
    ]),
])

# Define the callback
@callback(
    Output(component_id='valuator-output', component_property='children'),
    Input(component_id='submit-button', component_property='n_clicks'),
    State(component_id='valuator-input', component_property='value')
)
def update_output_div(n_clicks, domain_prefix):
    if n_clicks == 0:
        return "Please enter a domain prefix and click Submit."
    
    if not domain_prefix:
        return "Please enter a domain prefix."
    
    if '.' in domain_prefix:
        return "Invalid input. Please enter a valid domain prefix without a '.' character."
    
    domain = f"{domain_prefix}.box"
    domain_df = pd.DataFrame({'domain': [domain]})
    domain_processed = model_prep(domain_df)
    domain_value = value_domain(domain_processed)
    return html.Div([
        html.Div(f'Domain: {domain}', style={'font-weight': 'bold'}),
        html.Div(f'Estimated Value: ${round(domain_value, 2):,.2f}')
    ])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
