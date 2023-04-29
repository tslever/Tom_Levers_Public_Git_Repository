from dash import Dash, html, dcc, Output, Input
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
    'sex',
    'educ',
    'region',
    'age',
    'coninc',
    'prestg10',
    'mapres10',
    'papres10',
    'sei10',
    'satjob',
    'fechld',
    'fefam',
    'fepol',
    'fepresch',
    'meovrwrk'
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

text = '''
According to the [Pew Research Center](https://www.pewresearch.org/social-trends/2023/03/01/the-enduring-grip-of-the-gender-pay-gap/), "Women generally begin their careers closer to wage parity with men, but they lose ground as they age and progress through their work lives, a pattern that has remained consistent over time". According to [Forbes Magazine](https://www.forbes.com/advisor/business/gender-pay-gap-statistics/), "There are two types of gender pay gaps: the controlled and uncontrolled gap. The controlled gap measures the difference in pay between men and women performing the same job, with the same experience and qualifications. The uncontrolled gap represents the overall difference in pay between men and women, considering all the jobs and industries in which they work... When comparing women and men with the same job title, seniority level and hours worked, a gender gap of 11% still exists in terms of take-home pay."

In order to study the gender wage gap, we consider data from the General Social Survey (GSS). According to the [National Opinion Research Center](https://gss.norc.org/About-The-GSS), "For five decades, the General Social Survey (GSS) has studied the growing complexity of American society. It is the only full-probability, personal-interview survey designed to monitor changes in both social characteristics and attitudes currently being conducted in the United States. The General Social Survey (GSS) is a nationally representative survey of adults in the United States conducted since 1972. The GSS collects data on contemporary American society in order to monitor and explain trends in opinions, attitudes and behaviors. The GSS has adapted questions from earlier surveys, thereby allowing researchers to conduct comparisons for up to 80 years. The GSS contains a standard core of demographic, behavioral, and attitudinal questions, plus topics of special interest. Among the topics covered are civil liberties, crime and violence, intergroup tolerance, morality, national spending priorities, psychological well-being, social mobility, and stress and traumatic events... The data come from the General Social Surveys, interviews administered to NORC national samples using a standard questionnaire." The data for this study include values for sex, years of formal education, personal annual income, occupational prestige, index of socioeconomic status, and agreement with, "It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family."
'''
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
    'Agreement With "A working mother can establish just as warm and secure a relationship with her children as a mother who does not work."': 'relationship',
    'Agreement With "It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family."': 'male_breadwinner',
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
        html.H2('Introduction'),
        dcc.Markdown(children = text),
        html.H2('Socioeconomic Success By Sex'),
        dcc.Graph(figure = table),
        dcc.Graph(
            id = 'bar_plot'#, # Callback step 5: The output from the callback gets sent here.
        ),
        html.H2('Select feature for which to create a bar plot.'),
        dcc.Dropdown(
            id = 'dropdown_menu_for_feature_whose_categories_forms_the_basis_of_bars', # Callback step 1: Just provide an ID for the element to be input to the callback.
            options = list(description_to_feature_whose_categories_forms_the_basis_of_bars.keys()),
            value = 'Job Satisfaction'
        ),
        html.H2('Select feature by which to group bars.'),
        dcc.Dropdown(
            id = 'dropdown_menu_for_feature_by_which_to_group',
            options = list(description_to_feature_by_which_to_group.keys()),
            value = 'None'
        ),
        html.H2('Annual Income vs. Occupational Prestige'),
        dcc.Graph(figure = scatter_plot),
        html.H2('Distributions Of Income By Sex And Occupational Prestige'),
        dcc.Graph(figure = facet_grid),
        html.H2('Distributions of Income and Occupation Prestige By Sex'),
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

# Callbacks go between the layout and the dashboard run commands.
# What I'm about to type is going to decorate the function that I type next.
@app.callback(
    # Callback step 4: Define output.
    Output(
        component_id = 'bar_plot',
        component_property = 'figure'
    ),
    # Callback step 2: Send the input to this callback.
    Input(
        component_id = 'dropdown_menu_for_feature_whose_categories_forms_the_basis_of_bars', # Look for the input to come from the feature with ID places.
        component_property = 'value'
    ),
    Input(
        component_id = 'dropdown_menu_for_feature_by_which_to_group',
        component_property = 'value'
    )
)
# Callback step 3: The callback sends the input to this function. Then it sends the function's output back to the callback.
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