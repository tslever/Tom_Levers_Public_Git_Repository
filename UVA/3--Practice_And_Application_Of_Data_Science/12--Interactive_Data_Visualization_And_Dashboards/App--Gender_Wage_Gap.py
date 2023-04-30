from dash import Dash, html, dcc, Output, Input, ctx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

gss = pd.read_csv(
    "https://github.com/jkropko/DS-6001/raw/master/localdata/gss2018.csv",
    encoding = 'cp1252',
    na_values = [
        'IAP',
        'IAP,DK,NA,uncodeable',
        'NOT SURE',
        'DK',
        'IAP, DK, NA, uncodeable',
        '.a',
        "CAN'T CHOOSE"
    ]
)
columns_of_interest = [
    'id',
    'wtss',
    'sex', # sex
    'educ', # education
    'region', # region
    'age',
    'coninc', # personal annual income
    'prestg10', # occupational prestige
    'mapres10',
    'papres10',
    'sei10', # index of socioeconomic status
    'satjob', # job satisfaction
    'fechld', # agreement with "A working mother can establish just as warm and secure a relationship with her children as a mother who does not work."
    'fefam', # agreement with "It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family."
    'fepol', # agreement with "Most men are better suited emotionally for politics than are most women."
    'fepresch', # "A preschool child is likely to suffer if his or her mother works."
    'meovrwrk' # "Family life often suffers because men concentrate too much on their work."
]
gss_clean = gss[columns_of_interest]
gss_clean = gss_clean.rename(
    {
        'wtss': 'weight',
        'educ': 'education',
        'coninc': 'income',
        'prestg10': 'job_prestige',
        'mapres10': 'mother_job_prestige',
        'papres10': 'father_job_prestige',
        'sei10': 'socioeconomic_index',
        'fechld': 'relationship',
        'fefam': 'male_breadwinner',
        'fehire': 'hire_women',
        'fejobaff': 'preference_hire_women',
        'fepol': 'men_bettersuited',
        'fepresch': 'child_suffer',
        'meovrwrk': 'men_overwork'
    },
    axis = 1
)
gss_clean.age = gss_clean.age.replace(
    {
        '89 or older': '89'
    }
)
gss_clean.age = gss_clean.age.astype('float')

paragraph_1 = 'According to the [Pew Research Center](https://www.pewresearch.org/social-trends/2023/03/01/the-enduring-grip-of-the-gender-pay-gap/), "Women generally begin their careers closer to wage parity with men, but they lose ground as they age and progress through their work lives, a pattern that has remained consistent over time". According to [Forbes Magazine](https://www.forbes.com/advisor/business/gender-pay-gap-statistics/), "There are two types of gender pay gaps: the controlled and uncontrolled gap. The controlled gap measures the difference in pay between men and women performing the same job, with the same experience and qualifications. The uncontrolled gap represents the overall difference in pay between men and women, considering all the jobs and industries in which they work... When comparing women and men with the same job title, seniority level and hours worked, a gender gap of 11% still exists in terms of take-home pay."'
paragraph_2 = 'In order to study the gender wage gap, we consider data from the General Social Survey (GSS). According to the [National Opinion Research Center](https://gss.norc.org/About-The-GSS), "For five decades, the General Social Survey (GSS) has studied the growing complexity of American society. It is the only full-probability, personal-interview survey designed to monitor changes in both social characteristics and attitudes currently being conducted in the United States. The General Social Survey (GSS) is a nationally representative survey of adults in the United States conducted since 1972. The GSS collects data on contemporary American society in order to monitor and explain trends in opinions, attitudes and behaviors. The GSS has adapted questions from earlier surveys, thereby allowing researchers to conduct comparisons for up to 80 years. The GSS contains a standard core of demographic, behavioral, and attitudinal questions, plus topics of special interest. Among the topics covered are civil liberties, crime and violence, intergroup tolerance, morality, national spending priorities, psychological well-being, social mobility, and stress and traumatic events... The data come from the General Social Surveys, interviews administered to NORC national samples using a standard questionnaire." The data for this study include values for sex, years of formal education, region, personal annual income, occupational prestige, index of socioeconomic status, job satisfaction, and agreement with five statements relating to gender roles.'
data_frame = (gss_clean
    .drop(
        columns = [
            'region',
            'satjob',
            'relationship',
            'male_breadwinner',
            'men_bettersuited',
            'child_suffer',
            'men_overwork'
        ]
    )
    .groupby(
        [
            'sex'
        ]
    )
    .mean()
)
data_frame = data_frame[['income', 'job_prestige', 'socioeconomic_index', 'education']]
data_frame = data_frame.round(2)
data_frame = data_frame.rename(
    columns = {
        'income': 'mean annual income',
        'job_prestige': 'occupational prestige',
        'socioeconomic_index': 'index of socioeconomic status',
        'education': 'years of education'
    }
)
data_frame = data_frame.reset_index()
table = ff.create_table(data_frame)
salmon = '#FA8072'
scatter_plot = px.scatter(
    gss_clean,
    x = 'job_prestige',
    y = 'income',
    color = 'sex',
    color_discrete_map = {'male': 'blue', 'female': salmon},
    trendline = 'ols',
    labels = {
        'job_prestige':'occupational prestige', 
        'income':'annual income'
    },
    hover_data = ['education', 'socioeconomic_index']
)
slice = gss_clean[['income', 'sex', 'job_prestige']]
slice['job_prestige_binned'] = pd.cut(
    slice.job_prestige,
    bins = 6
)
slice = slice.dropna()
slice = slice.sort_values(
    by = [
        'job_prestige_binned'
    ]
)
facet_grid = px.box(
    slice,
    x = 'income',
    y = 'sex',
    color = 'sex',
    color_discrete_map = {
        'male': 'blue',
        'female': salmon
    },
    facet_col = 'job_prestige_binned',
    facet_col_wrap = 2,
    labels = {
        'income': 'annual income'
    },
    height = 500
)
facet_grid = facet_grid.for_each_annotation(lambda a: a.update(text=a.text.replace("job_prestige_binned=", "Occupational Prestige: ")))
distributions_of_income_by_sex = px.box(
    gss_clean,
    x = 'sex',
    y = 'income',
    color = 'sex',
    color_discrete_map = {
        'male': 'blue',
        'female': salmon
    },
    labels = {
        'income': 'annual income'
    },
    width = 600,
    height = 600
)
distributions_of_income_by_sex = distributions_of_income_by_sex.update_layout(
    xaxis_title = None
)
distributions_of_income_by_sex = distributions_of_income_by_sex.update_layout(
    showlegend = False
)
distributions_of_occupational_prestige_by_sex = px.box(
    gss_clean,
    x = 'sex',
    y = 'job_prestige',
    color = 'sex',
    color_discrete_map = {'male': 'blue', 'female': salmon},
    labels = {'job_prestige':'ocuupational prestige'},
    width = 600,
    height = 600
)
distributions_of_occupational_prestige_by_sex = distributions_of_occupational_prestige_by_sex.update_layout(
    xaxis_title = None
)
distributions_of_occupational_prestige_by_sex = distributions_of_occupational_prestige_by_sex.update_layout(
    showlegend = False
)

description_to_feature_whose_categories_forms_the_basis_of_bars = {
    'Job Satisfaction': 'satjob',
    'Agreement With "A working mother can establish just as warm and secure a relationship with her children\nas a mother who does not work."': 'relationship',
    'Agreement With "It is much better for everyone involved if the man is the achiever outside the home\nand the woman takes care of the home and family."': 'male_breadwinner',
    'Agreement With "Most men are better suited emotionally for politics than are most women."': 'men_bettersuited',
    'Agreement With "A preschool child is likely to suffer if his or her mother works."': 'child_suffer',
    'Agreement With "Family life often suffers because men concentrate too much on their work."': 'men_overwork'
}
description_to_feature_by_which_to_group = {
    'None': 'None',
    'Sex': 'sex',
    'Region': 'region',
    'Years Of Education': 'education_binned'
}
feature_whose_categories_forms_the_basis_of_bars_to_bar_plot_horizontal_axis_label = {
    'satjob': 'satisfaction',
    'relationship': 'agreement',
    'male_breadwinner': 'agreement',
    'men_bettersuited': 'agreement',
    'child_suffer': 'agreement',
    'men_overwork': 'agreement'
}
gss_clean['education_binned'] = pd.cut(gss_clean.education, bins = [-0.1, 10, 11, 12, 13, 14, 15, 16, np.Inf], labels = ['10 years or fewer', '11 years', '12 years', '13 years', '14 years', '15 years', '16 years', 'More than 16 years'])
features_with_horizontal_axis_label_agreement = ['relationship', 'male_breadwinner', 'men_bettersuited', 'child_suffer', 'men_overwork']
gss_clean['satjob'] = gss_clean['satjob'].astype('category')
gss_clean['satjob'] = (gss_clean['satjob']
    .cat
    .reorder_categories(
        [
            'very satisfied',
            'mod. satisfied',
            'a little dissat',
            'very dissatisfied'
        ]
    )
)
gss_clean['relationship'] = gss_clean['relationship'].astype('category')
gss_clean['relationship'] = (gss_clean['relationship']
    .cat
    .reorder_categories(
        [
            'strongly agree',
            'agree',
            'disagree',
            'strongly disagree'
        ]
    )
)
gss_clean['male_breadwinner'] = gss_clean['male_breadwinner'].astype('category')
gss_clean['male_breadwinner'] = (gss_clean['male_breadwinner']
    .cat
    .reorder_categories(
        [
            'strongly agree',
            'agree',
            'disagree',
            'strongly disagree'
        ]
    )
)
gss_clean['men_bettersuited'] = gss_clean['men_bettersuited'].astype('category')
gss_clean['men_bettersuited'] = (gss_clean['men_bettersuited']
    .cat
    .reorder_categories(
        [
            'agree',
            'disagree'
        ]
    )
)
gss_clean['child_suffer'] = gss_clean['child_suffer'].astype('category')
gss_clean['child_suffer'] = (gss_clean['child_suffer']
    .cat
    .reorder_categories(
        [
            'strongly agree',
            'agree',
            'disagree',
            'strongly disagree'
        ]
    )
)
gss_clean['men_overwork'] = gss_clean['men_overwork'].astype('category')
gss_clean['men_overwork'] = (gss_clean['men_overwork']
    .cat
    .reorder_categories(
        [
            'strongly agree',
            'agree',
            'neither agree nor disagree',
            'disagree',
            'strongly disagree'
        ]
    )
)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets = external_stylesheets)
app.layout = html.Div(
    [
        html.H1('Gender Wage Gap'),
        html.Table(
            [
                html.Tr(
                    [
                        html.Td(
                            [
                                html.Button(
                                    'Introduction',
                                    id = 'button_labeled_Introduction',
                                    style = {'width': '100%'}
                                ),
                                html.Br(),
                                html.Button(
                                    'Socioeconomic Success By Sex',
                                    id = 'button_labeled_Socioeconomic_Success_By_Sex',
                                    style = {'width': '100%'}
                                ),
                                html.Br(),
                                html.Button(
                                    'Interactive Bar Plot',
                                    id = 'button_labeled_Interactive_Bar_Plot',
                                    style = {'width': '100%'}
                                ),
                                html.Br(),
                                html.Button(
                                    'Annual Income Versus Occupational Prestige',
                                    id = 'button_labeled_Annual_Income_Versus_Occupational_Prestige',
                                    style = {'width': '100%'}
                                ),
                                html.Br(),
                                html.Button(
                                    'Distributions Of Income By Sex And Occupational Prestige',
                                    id = 'button_labeled_Distributions_Of_Income_By_Sex_And_Occupational_Prestige',
                                    style = {'width': '100%'}
                                ),
                                html.Br(),
                                html.Button(
                                    'Distributions Of Income And Occupation Prestige By Sex',
                                    id = 'button_labeled_Distributions_Of_Income_And_Occupational_Prestige_By_Sex',
                                    style = {'width': '100%'}
                                )
                            ],
                            rowSpan = 2,
                            style = {
                                'width': '33%',
                                'border': 'solid'
                            }
                        ),
                        html.Td(
                            html.Div(
                                id = 'title_of_figure'
                            ),
                            style = {
                                'border': 'solid'
                            }
                        )
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            [
                                html.Div(
                                    id = 'figure'
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id = 'bar_plot'
                                        ),
                                        'Select feature for which to create a bar plot.',
                                        dcc.Dropdown(
                                            id = 'dropdown_menu_for_feature_whose_categories_forms_the_basis_of_bars',
                                            options = list(description_to_feature_whose_categories_forms_the_basis_of_bars.keys()),
                                            value = 'Job Satisfaction'
                                        ),
                                        html.Br(),
                                        'Select feature by which to group bars.',
                                        dcc.Dropdown(
                                            id = 'dropdown_menu_for_feature_by_which_to_group',
                                            options = list(description_to_feature_by_which_to_group.keys()),
                                            value = 'None'
                                        )
                                    ],
                                    style = {'display': 'none'}
                                )
                            ],
                            style = {
                                'border': 'solid'
                            }
                        )
                    ]
                )
            ],
            style = {'width': '100%'}
        )
    ]
)

@app.callback(
    Output(
        component_id = 'title_of_figure',
        component_property = 'children'
    ),
    Input(
        'button_labeled_Introduction',
        'n_clicks'
    ),
    Input(
        'button_labeled_Socioeconomic_Success_By_Sex',
        'n_clicks'
    ),
    Input(
        'button_labeled_Interactive_Bar_Plot',
        'n_clicks'
    ),
    Input(
        'button_labeled_Annual_Income_Versus_Occupational_Prestige',
        'n_clicks'
    ),
    Input(
        'button_labeled_Distributions_Of_Income_By_Sex_And_Occupational_Prestige',
        'n_clicks'
    ),
    Input(
        'button_labeled_Distributions_Of_Income_And_Occupational_Prestige_By_Sex',
        'n_clicks'
    )
)
def generate_title_for_figure(Introduction, Socioeconomic_Success_By_Sex, Bar_Plot, Annual_Income_Versus_Occupational_Prestige, Distributions_Of_Income_By_Sex_And_Occupational_Prestige, Distributions_Of_Income_And_Occupational_Prestige_By_Sex):
    button_clicked = ctx.triggered_id
    print(button_clicked)
    if (button_clicked == None) or (button_clicked == 'button_labeled_Introduction'):
        return 'Introduction'
    elif button_clicked == 'button_labeled_Socioeconomic_Success_By_Sex':
        return 'Socioeconomic Success By Sex'
    elif button_clicked == 'button_labeled_Interactive_Bar_Plot':
        return 'Interactive Bar Plot'
    elif button_clicked == 'button_labeled_Annual_Income_Versus_Occupational_Prestige':
        return 'Annual Income Versus Occupational Prestige'
    elif button_clicked == 'button_labeled_Distributions_Of_Income_By_Sex_And_Occupational_Prestige':
        return 'Distributions Of Income By Sex And Occupational Prestige'
    elif button_clicked == 'button_labeled_Distributions_Of_Income_And_Occupational_Prestige_By_Sex':
        return 'Distributions Of Income And Occupational Prestige By Sex'
    else:
        return 'Placeholder'
    
@app.callback(
    Output(
        component_id = 'figure',
        component_property = 'children'
    ),
    Input(
        'button_labeled_Introduction',
        'n_clicks'
    ),
    Input(
        'button_labeled_Socioeconomic_Success_By_Sex',
        'n_clicks'
    ),
    Input(
        'button_labeled_Interactive_Bar_Plot',
        'n_clicks'
    ),
    Input(
        'button_labeled_Annual_Income_Versus_Occupational_Prestige',
        'n_clicks'
    ),
    Input(
        'button_labeled_Distributions_Of_Income_By_Sex_And_Occupational_Prestige',
        'n_clicks'
    ),
    Input(
        'button_labeled_Distributions_Of_Income_And_Occupational_Prestige_By_Sex',
        'n_clicks'
    )
)
def generate_figure(Introduction, Socioeconomic_Success_By_Sex, Bar_Plot, Annual_Income_Versus_Occupational_Prestige, Distributions_Of_Income_By_Sex_And_Occupational_Prestige, Distributions_Of_Income_And_Occupational_Prestige_By_Sex):
    button_clicked = ctx.triggered_id
    print(button_clicked)
    if (button_clicked == None) or (button_clicked == 'button_labeled_Introduction'):
        return html.Div(
            [
                dcc.Markdown(paragraph_1),
                dcc.Markdown(paragraph_2)
            ]
        )
    elif button_clicked == 'button_labeled_Socioeconomic_Success_By_Sex':
        return html.Div(
            dcc.Graph(figure = table)
        )
    elif button_clicked == 'button_labeled_Interactive_Bar_Plot':
        return html.Div(
            [
                dcc.Graph(
                    id = 'bar_plot'
                ),
                'Select feature for which to create a bar plot.',
                dcc.Dropdown(
                    id = 'dropdown_menu_for_feature_whose_categories_forms_the_basis_of_bars',
                    options = list(description_to_feature_whose_categories_forms_the_basis_of_bars.keys()),
                    value = 'Job Satisfaction'
                ),
                html.Br(),
                'Select feature by which to group bars.',
                dcc.Dropdown(
                    id = 'dropdown_menu_for_feature_by_which_to_group',
                    options = list(description_to_feature_by_which_to_group.keys()),
                    value = 'None'
                )
            ]
        )
    elif button_clicked == 'button_labeled_Annual_Income_Versus_Occupational_Prestige':
        return dcc.Graph(figure = scatter_plot)
    elif button_clicked == 'button_labeled_Distributions_Of_Income_By_Sex_And_Occupational_Prestige':
        return dcc.Graph(figure = facet_grid)
    elif button_clicked == 'button_labeled_Distributions_Of_Income_And_Occupational_Prestige_By_Sex':
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(figure = distributions_of_income_by_sex)
                    ],
                    style = {
                        'width': '48%',
                        'float': 'left'
                    }
                ),
                html.Div(
                    [            
                        dcc.Graph(figure = distributions_of_occupational_prestige_by_sex)
                    ],
                    style = {
                        'width': '48%',
                        'float': 'right'
                    }
                )
            ]
        )
    else:
        return 'Placeholder'

@app.callback(
    Output(
        component_id = 'bar_plot',
        component_property = 'figure'
    ),
    Input(
        component_id = 'dropdown_menu_for_feature_whose_categories_forms_the_basis_of_bars',
        component_property = 'value'
    ),
    Input(
        component_id = 'dropdown_menu_for_feature_by_which_to_group',
        component_property = 'value'
    )
)
def generate_bar_plot(description_of_feature_whose_categories_form_the_basis_of_bars, description_of_feature_by_which_to_group):
    feature_whose_categories_forms_the_basis_of_bars = description_to_feature_whose_categories_forms_the_basis_of_bars[description_of_feature_whose_categories_form_the_basis_of_bars]
    horizontal_axis_label = feature_whose_categories_forms_the_basis_of_bars_to_bar_plot_horizontal_axis_label[feature_whose_categories_forms_the_basis_of_bars]
    feature_by_which_to_group = description_to_feature_by_which_to_group[description_of_feature_by_which_to_group]
    if (feature_by_which_to_group == 'None'):
        grouped_value_counts = (gss_clean
            .groupby(
                [feature_whose_categories_forms_the_basis_of_bars]
            )
            .size()
        ).reset_index().rename(columns = {feature_whose_categories_forms_the_basis_of_bars: horizontal_axis_label, 0: 'number of people'})
        bar_plot = px.bar(grouped_value_counts, x = horizontal_axis_label, y = 'number of people', title = description_of_feature_whose_categories_form_the_basis_of_bars)
    else:
        grouped_value_counts = gss_clean.groupby([feature_whose_categories_forms_the_basis_of_bars, feature_by_which_to_group]).size().reset_index().rename(columns = {feature_whose_categories_forms_the_basis_of_bars: horizontal_axis_label, 0: 'number of people'})
        bar_plot = px.bar(grouped_value_counts, x = horizontal_axis_label, y = 'number of people', title = description_of_feature_whose_categories_form_the_basis_of_bars, color = feature_by_which_to_group, barmode = 'group')
    return bar_plot

if __name__ == '__main__':
    app.run_server(debug = True, port = 8051)