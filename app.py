import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output


import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import statsmodels as sm
import plotly.figure_factory as ff
import numpy as np
import math
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

# Load data
df_nasa = pd.read_csv('NASA Appropriations.csv')
df_vc = pd.read_csv('Space Data Clean.csv')

#SELECTING WHICH VERTICALS TO ANALYZE
# print(df_vc.head())

vertical_comprehensive_list = []

for industry in df_vc['Verticals']:
    vertical_comprehensive_list.append(industry.split(","))

for name in vertical_comprehensive_list:
    for item in range(len(name)):
        name[item] = name[item].replace(" ","")
        
vertical_comprehensive_list_no_repeats = []

for name in vertical_comprehensive_list:
    for item in name:
        if item not in vertical_comprehensive_list_no_repeats:
            vertical_comprehensive_list_no_repeats.append(item)
    
# print(vertical_comprehensive_list_no_repeats)

verticals_of_interest = ['Industrials','Manufacturing','Space Technology','Oil & Gas','MobilityTech','CleanTech','RoboticsandDrone','Big Data','AdvancedManufacturing','Infrastructure','AI&ML','IoT','AgTech','Cybersecurity','Autonomous Vehicles','VR']

# us_space_startup = []

# for startups founded from year 1990


def us_commercial_space(data):
    us_space_startups = []
    for ind, row in data.iterrows():
        if row[11] >= 1990 and row[15] == 'United States':
            us_space_startups.append(row[11])
    return us_space_startups

us_startup_data = us_commercial_space(df_vc)

def year_count(data):
    year_count_list = []
    for year in range(1990,2020):
            year_count_count = data.count(year)
            year_count_list.append(year_count_count)
    return year_count_list

us_startup_count_by_year = year_count(us_startup_data)


# NASA APPROPRIATIONS VS. PRIVATE DATA
#Comparing NASA budget to U.S. space startups founded that year

# Create figure with secondary y-axis
fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig1.add_trace(
    go.Scatter(x=df_nasa['Year'], y=df_nasa['NASA Appopriation (B)'], name="NASA Appropriation", marker_color = 'midnightblue'),
    secondary_y=False,
)

fig1.add_trace(
    go.Scatter(x=df_nasa['Year'], y=us_startup_count_by_year, name="Startups", marker_color = 'maroon'),
    secondary_y=True,
)

# Add figure title
fig1.update_layout(
    title_text="NASA Budget vs. Startups Founded"
)

# Set x-axis title
fig1.update_xaxes(title_text="Year")

# Set y-axes titles
fig1.update_yaxes(title_text="NASA Appropriation (B)", secondary_y=False)
fig1.update_yaxes(title_text="Number of Startups <br> Founded", secondary_y=True)


NASA_appropriation_pct_change = (df_nasa['NASA Appopriation (B)'].pct_change())*100
# print(NASA_appropriation_pct_change)

us_startup_count_by_year_pct_change = []

for i in range(len(us_startup_count_by_year)):
    if i == 0:
        new_value = 0
        us_startup_count_by_year_pct_change.append(new_value)
    else:
        new_value = ((us_startup_count_by_year[i]-us_startup_count_by_year[i-1])/us_startup_count_by_year[i-1])*100
        us_startup_count_by_year_pct_change.append(new_value)
        
# print(us_startup_count_by_year)
# print(us_startup_count_by_year_pct_change)

#Comparing NASA budget to U.S. space startups founded that year

# Create figure with secondary y-axis
fig2 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig2.add_trace(
    go.Scatter(x=df_nasa['Year'], y=NASA_appropriation_pct_change, name="NASA Appropriation", marker_color = 'midnightblue'),
    secondary_y=False,
)

fig2.add_trace(
    go.Scatter(x=df_nasa['Year'], y=us_startup_count_by_year_pct_change, name="Startups", marker_color = 'maroon'),
    secondary_y=True,
)

# Add figure title
fig2.update_layout(
    title_text="NASA Budget vs. Number of Startups Founded Growth"
)

# Set x-axis title
fig2.update_xaxes(title_text="Year")

# Set y-axes titles
fig2.update_yaxes(title_text='NASA  Appropriation Growth Y/Y', secondary_y=False)
fig2.update_yaxes(title_text='% Change Number <br> of Startups Founded Y/Y', secondary_y=True)

fig3 = px.bar(df_nasa, x=df_nasa['Year'], y=df_nasa['NASA Appopriation (B)'],
             hover_data=[df_nasa['% of Federal Budget']], color=df_nasa['Party'],color_discrete_sequence=["maroon", "midnightblue"],
             title = 'NASA Budget by Party', height=400)

# fig.update_traces(marker_color=['red','blue'])

# fig3.show()

fig4 = px.bar(df_nasa, x=df_nasa['Year'], y=df_nasa['NASA Appropriation Adjusted for Inflation'],
             hover_data=[df_nasa['% of Federal Budget']], color=df_nasa['Party'],color_discrete_sequence=["maroon", "midnightblue"],
             title = 'NASA Budget by Party <br>(Adjusted for Inflation)', height=400)

fig4.update_yaxes(title_text='NASA  Appropriation (B)', secondary_y=False)

# fig4.show()

# VERTICALS CLASS

class Verticals:
    def __init__(self,name,data):
        self.name = name
        self.data = self.vert_data(name,data)
        
    def vert_data(self,name,data):
        vertical_data_list = []
        for ind, row in data.iterrows():
            if name in row[1]:
                new_data_dict = dict(zip(list(df_vc.columns.values),row.values.tolist()))
                vertical_data_list.append(new_data_dict)
        vertical_data = pd.DataFrame(vertical_data_list)      
        return vertical_data
    
    def country_breakdown(self):
        fig = px.pie(self.data, values='Count', names='HQ Country', title= self.name + ' Country Breakdown',color_discrete_sequence=px.colors.sequential.RdBu)
        # fig.show()
        return fig
        
    def found_to_round(self):
        #make en empty list
        difference_found_to_round = []
        #take difference of founding year vs.first financing year
        for i in range(len(self.data['Company Name'])):
            if math.isnan(self.data['First Financing Deal Year'][i]) or math.isnan(self.data['Year Founded'][i]):
                continue
            difference = self.data['First Financing Deal Year'][i] - self.data['Year Founded'][i]
            # add to a list
            difference_found_to_round.append(difference)
        #create distribution graph
        fig = ff.create_distplot([(difference_found_to_round)],['Time Between Year Founded and First Financing Year'],colors = ['olive'],show_rug=False)
        fig.update_xaxes(title="Years")
        fig.update_yaxes(title="Probability")
        # fig.show()
        return fig

    def valuation_distr(self):
#         fig = px.histogram(self.data,x='Last Known Valuation (M)',title = 'Valuation Across ' + self.name + ' Companies', nbins = 150)
        valuations = []
        for i in range(len(self.data['Company Name'])):
            if math.isnan(self.data['Last Known Valuation (M)'][i]):
                continue
            valuations.append(self.data['Last Known Valuation (M)'][i])
        fig = ff.create_distplot([valuations],['Valuation Across ' + self.name + ' Companies'],bin_size = 20,colors = ['darkolivegreen'],show_rug=False)
        fig.update_xaxes(title="Valuation")
        fig.update_yaxes(title="Probability")
        # fig.show()
        return fig

    def ownership_status(self):
        fig = px.pie(self.data, values='Count', names='Ownership Status', title= self.name + ' Ownership Status',color_discrete_sequence=px.colors.sequential.RdBu)
        # fig.show()
        return fig
        
    def business_status(self):
        fig = px.pie(self.data, values='Count', names='Business Status', title= self.name + ' Business Status',color_discrete_sequence=px.colors.sequential.RdBu)
        # fig.show()
        return fig
        
    def age_vs_capital_raised(self):
        age = []
        for i in range(len(self.data['Company Name'])):
            difference = self.data['Current Year'][i] - self.data['Year Founded'][i]
            # add to a list
            age.append(difference)
        fig = px.scatter(x=age, y=self.data['Total Raised'],title= 'Company Age vs. Total Raised',color_discrete_sequence=px.colors.sequential.RdBu,trendline="ols")
        fig.update_xaxes(title="Age of Company (Years)")
        fig.update_yaxes(title="Reported Amount Raised (Millions)")
        # fig.show()
        return fig
        
    def highest_valued(self):
        self.data['Last Known Valuation (M)'] = self.data['Last Known Valuation (M)'].astype(float)
        return self.data.nlargest(5,['Last Known Valuation (M)'])
    
    def most_raised(self):
        self.data['Total Raised'] = self.data['Total Raised'].astype(float)
        return self.data.nlargest(5,['Total Raised'])

#Objects
Industrials = Verticals('Industrials',df_vc)
Manufacturing = Verticals('Manufacturing',df_vc)
Space_Technology = Verticals('Space Technology',df_vc)
Oil_and_Gas = Verticals('Oil & Gas',df_vc)
Mobility_Tech = Verticals('Mobility Tech',df_vc)
CleanTech = Verticals('CleanTech',df_vc)
Robotics_and_Drones = Verticals('Robotics and Drones',df_vc)
Big_Data = Verticals('Big Data',df_vc)
Advanced_Manufacturing = Verticals('Advanced Manufacturing',df_vc)
Infrastructure = Verticals('Infrastructure',df_vc)
AI_and_ML = Verticals('Artificial Intelligence & Machine Learning',df_vc)
IoT = Verticals('Internet of Things',df_vc)
AgTech = Verticals('AgTech',df_vc)

# Initialize the app
app = dash.Dash(__name__)
# app.config.suppress_callback_exceptions = True

# colors = {'background':'#FFF8DC','text':'#7FDBFF'}

app.layout = html.Div([
    html.H1(children="Alaina's Space Dash Board"),
    html.Div([
        html.Div([
            # html.H3('Column 1'),
            dcc.Graph(id='g1',figure=fig1)
        ], className="six columns"),
        html.Div([
            # html.H3('Column 2'),
            dcc.Graph(id='g2',figure=fig2)
        ], className="six columns")
    ], className="row"),
    html.Div([
        html.Div([
            # html.H3('Column 3'),
            dcc.Graph(id='g3',figure=fig3)
        ], className="six columns"),
        html.Div([
            # html.H3('Column 4'),
            dcc.Graph(id='g4',figure=fig4)
        ], className="six columns")
    ], className="row"),
    # dcc.Graph(id='thisone',figure=fig1),
    
    # dcc.Slider(
    #     id='year-slider',
    #     min=df['year'].min(),
    #     max=df['year'].max(),
    #     value=df['year'].min(),
    #     marks={str(year): str(year) for year in df['year'].unique()},
    #     step=None
    # )
    dcc.Dropdown(id='vert-dropdown', options=[{'label':'Manufacturing','value':'Manufacturing'},
                                                    {'label':'Industrials','value':'Industrials'},
                                                    {'label':'Space Technology','value':'Space Technology'},
                                                    {'label':'Oil and Gas','value':'Oil and Gas'},
                                                    {'label':'Mobility Tech','value':'Mobility Tech'},
                                                    {'label':'Clean Tech','value':'Clean Tech'},
                                                    {'label':'Robotics and Drones','value':'Robotics and Drones'},
                                                    {'label':'Big Data','value':'Big Data'},
                                                    {'label':'Advanced Manufacturing','value':'Advanced Manufacturing'},
                                                    {'label':'Infrastructure','value':'Infrastructure'},
                                                    {'label':'AI/ML','value':'AI/ML'},
                                                    {'label':'IoT','value':'IoT'},
                                                    {'label':'AgTech','value':'AgTech'}],
                                                    multi=False, value='Manufacturing',
                                                    # style={'backgroundColor': '#1E1E1E'},
                                                    className='vertical'
                                                    ),
    dcc.Dropdown(id='g-type',options=[{'label':'Country Breakdown','value':'Country Breakdown'},
        {'label':'Found 2 Round','value':'Found 2 Round'},
        # {'label':'Valuation Distribution','value':'Valuation Distribution'},
        {'label':'Ownership Status','value':'Ownership Status'},
        {'label':'Business Status','value':'Business Status'},
        {'label':'Age vs Capital Raised','value':'Age vs Capital Raised'}],
        # {'label':'Highest Valued','value':'Highest Valued'},
        # {'label':'Most Raised','value':'Most Raised'}],
        value='Country Breakdown'),
    dcc.Graph(id='tempfig')
])

# app.layout = html.Div(
#     children=[
#         html.Div(className='row',
#                  children=[
#                     html.Div(className='four columns div-user-controls',
#                              children=[
#                                  html.H2('SPACE DATA'),
#                                  html.P('Visualising the space economy with Plotly - Dash.'),
#                                  html.P('Pick a vertical to analyze.'),
#                                  html.Div(
#                                      className='div-for-dropdown',
#                                      children=[
#                                          dcc.Dropdown(id='vert-dropdown', options=[{'label':'Manufacturing','value':'Manufacturing'},
#                                                     {'label':'Industrials','value':'Industrials'}],
#                                                     multi=False, value='Manufacturing',
#                                                     style={'backgroundColor': '#1E1E1E'},
#                                                     className='vertical'
#                                                     ),
#                                      ],
#                                      style={'color': '#1E1E1E'})
#                                 ]
#                              ),
#                     html.Div(className='eight columns div-for-charts bg-grey',
#                              children=[
#                                  dcc.Graph(id='tempfig', config={'displayModeBar': False}, animate=True)
#                              ])
#                               ])
#         ]

# )


# # Callback for timeseries price
@app.callback(Output('tempfig', 'figure'),
              [Input('vert-dropdown', 'value'),
              Input('g-type', 'value')])
def update_graph(drop_val1,gval):
    print(drop_val1)
    obj1 = Industrials
    if drop_val1 == 'Manufacturing':
        # print('to mfg')
        obj1 = Manufacturing
    elif drop_val1 == 'Industrials':
        # print('to ind')
        obj1 = Industrials
    elif drop_val1 == 'Space Technology':
        obj1 = Space_Technology
    elif drop_val1 == 'Oil and Gas':
        obj1 = Oil_and_Gas
    elif drop_val1 == 'Mobility Tech':
        obj1 = Mobility_Tech
    elif drop_val1 == 'Clean Tech':
        obj1 = CleanTech
    elif drop_val1 == 'Robotics and Drones':
        obj1 = Robotics_and_Drones
    elif drop_val1 == 'Big Data':
        obj1 = Big_Data
    elif drop_val1 == 'Advanced Manufacturing':
        obj1 = Advanced_Manufacturing
    elif drop_val1 == 'Infrastructure':
        obj1 = Infrastructure
    elif drop_val1 == 'AI/ML':
        obj1 = AI_and_ML
    elif drop_val1 == 'IoT':
        obj1 = IoT
    elif drop_val1 == 'AgTech':
        obj1 = AgTech


    if gval == 'Country Breakdown':
        fig = obj1.country_breakdown()
    elif gval == 'Found 2 Round':
        fig = obj1.found_to_round()
    # elif gval == 'Valuation Distribution':
    #     fig = obj1.valuation_distr()
    elif gval == 'Ownership Status':
        fig = obj1.ownership_status()
    elif gval == 'Business Status':
        fig = obj1.business_status()
    elif gval == 'Age vs Capital Raised':
        fig = obj1.age_vs_capital_raised()
    elif gval == 'Highest Valued':
        fig = obj1.highest_valued()
    elif gval == 'Most Raised':
        fig = obj1.most_raised()


    # fig = obj1.country_breakdown()
    # fig.update_layout(plot_bgcolor=colors['background'],paper_bgcolor=colors['background'],
    #     font_color=colors['text'])
    fig.update_layout()
    return fig

    # trace1 = []
    # # df_sub = df
    # for stock in selected_dropdown_value:
    #     trace1.append(go.Scatter(x=df_sub[df_sub['stock'] == stock].index,
    #                              y=df_sub[df_sub['stock'] == stock]['value'],
    #                              mode='lines',
    #                              opacity=0.7,
    #                              name=stock,
    #                              textposition='bottom center'))
    # traces = [trace1]
    # data = [val for sublist in traces for val in sublist]
    # figure = {'data': data,
    #           'layout': go.Layout(
    #               colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
    #               template='plotly_dark',
    #               paper_bgcolor='rgba(0, 0, 0, 0)',
    #               plot_bgcolor='rgba(0, 0, 0, 0)',
    #               margin={'b': 15},
    #               hovermode='x',
    #               autosize=True,
    #               title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
    #               xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
    #           ),

    #           }

    # return figure


if __name__ == '__main__':
    app.run_server(debug=True)