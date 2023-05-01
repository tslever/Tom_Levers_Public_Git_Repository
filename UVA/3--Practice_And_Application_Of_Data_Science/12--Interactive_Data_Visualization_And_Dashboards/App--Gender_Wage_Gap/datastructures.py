import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

data_frame_of_data_of_General_Social_Survey = pd.read_csv(
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
# Jonathan Kropko chose columns of interest. The columns names with comments correspond to columns used in "Gender Wage Gap".
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
    'fepresch', # agreement with "A preschool child is likely to suffer if his or her mother works."
    'meovrwrk' # agreement with "Family life often suffers because men concentrate too much on their work."
]
data_frame_of_data_of_General_Social_Survey = data_frame_of_data_of_General_Social_Survey[columns_of_interest]
# Jonathan Kropko chose new column names.
data_frame_of_data_of_General_Social_Survey = data_frame_of_data_of_General_Social_Survey.rename(
    columns = {
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
    }
)
data_frame_of_data_of_General_Social_Survey['age'] = data_frame_of_data_of_General_Social_Survey['age'].replace(
    {
        '89 or older': '89'
    }
)
data_frame_of_data_of_General_Social_Survey['age'] = data_frame_of_data_of_General_Social_Survey['age'].astype('float')

data_frame_of_data_of_General_Social_Survey['education_binned'] = pd.cut(
    data_frame_of_data_of_General_Social_Survey.education,
    bins = [
        -0.1,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        np.Inf
    ],
    labels = [
        '10 years or fewer',
        '11 years',
        '12 years',
        '13 years',
        '14 years',
        '15 years',
        '16 years',
        'More than 16 years'
    ]
)
data_frame_of_data_of_General_Social_Survey['satjob'] = data_frame_of_data_of_General_Social_Survey['satjob'].astype('category')
data_frame_of_data_of_General_Social_Survey['satjob'] = (data_frame_of_data_of_General_Social_Survey['satjob']
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
data_frame_of_data_of_General_Social_Survey['relationship'] = data_frame_of_data_of_General_Social_Survey['relationship'].astype('category')
data_frame_of_data_of_General_Social_Survey['relationship'] = (data_frame_of_data_of_General_Social_Survey['relationship']
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
data_frame_of_data_of_General_Social_Survey['male_breadwinner'] = data_frame_of_data_of_General_Social_Survey['male_breadwinner'].astype('category')
data_frame_of_data_of_General_Social_Survey['male_breadwinner'] = (data_frame_of_data_of_General_Social_Survey['male_breadwinner']
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
data_frame_of_data_of_General_Social_Survey['men_bettersuited'] = data_frame_of_data_of_General_Social_Survey['men_bettersuited'].astype('category')
data_frame_of_data_of_General_Social_Survey['men_bettersuited'] = (data_frame_of_data_of_General_Social_Survey['men_bettersuited']
    .cat
    .reorder_categories(
        [
            'agree',
            'disagree'
        ]
    )
)
data_frame_of_data_of_General_Social_Survey['child_suffer'] = data_frame_of_data_of_General_Social_Survey['child_suffer'].astype('category')
data_frame_of_data_of_General_Social_Survey['child_suffer'] = (data_frame_of_data_of_General_Social_Survey['child_suffer']
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
data_frame_of_data_of_General_Social_Survey['men_overwork'] = data_frame_of_data_of_General_Social_Survey['men_overwork'].astype('category')
data_frame_of_data_of_General_Social_Survey['men_overwork'] = (data_frame_of_data_of_General_Social_Survey['men_overwork']
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

paragraph_1 = 'According to the [Pew Research Center](https://www.pewresearch.org/social-trends/2023/03/01/the-enduring-grip-of-the-gender-pay-gap/), "Women generally begin their careers closer to wage parity with men, but they lose ground as they age and progress through their work lives, a pattern that has remained consistent over time". According to [Forbes Magazine](https://www.forbes.com/advisor/business/gender-pay-gap-statistics/), "There are two types of gender pay gaps: the controlled and uncontrolled gap. The controlled gap measures the difference in pay between men and women performing the same job, with the same experience and qualifications. The uncontrolled gap represents the overall difference in pay between men and women, considering all the jobs and industries in which they work... When comparing women and men with the same job title, seniority level and hours worked, a gender gap of 11% still exists in terms of take-home pay."'
paragraph_2 = 'In order to study the gender wage gap, we consider data from the General Social Survey (GSS). According to the [National Opinion Research Center](https://gss.norc.org/About-The-GSS), "For five decades, the General Social Survey (GSS) has studied the growing complexity of American society. It is the only full-probability, personal-interview survey designed to monitor changes in both social characteristics and attitudes currently being conducted in the United States. The General Social Survey (GSS) is a nationally representative survey of adults in the United States conducted since 1972. The GSS collects data on contemporary American society in order to monitor and explain trends in opinions, attitudes and behaviors. The GSS has adapted questions from earlier surveys, thereby allowing researchers to conduct comparisons for up to 80 years. The GSS contains a standard core of demographic, behavioral, and attitudinal questions, plus topics of special interest. Among the topics covered are civil liberties, crime and violence, intergroup tolerance, morality, national spending priorities, psychological well-being, social mobility, and stress and traumatic events... The data come from the General Social Surveys, interviews administered to NORC national samples using a standard questionnaire." The data for this study include values for sex, years of formal education, region, personal annual income, occupational prestige, index of socioeconomic status, job satisfaction, and agreement with five statements relating to gender roles.'

table = (data_frame_of_data_of_General_Social_Survey
    .drop( # categorical
        columns = [
            'region',
            'satjob',
            'relationship',
            'male_breadwinner',
            'men_bettersuited',
            'child_suffer',
            'men_overwork',
            'education_binned'
        ]
    )
    .groupby(
        [
            'sex'
        ]
    )
    .mean()
)
table = table[['income', 'job_prestige', 'socioeconomic_index', 'education']]
table = table.round(2)
table = table.rename(
    columns = {
        'income': 'mean annual income',
        'job_prestige': 'occupational prestige',
        'socioeconomic_index': 'index of socioeconomic status',
        'education': 'years of education'
    }
)
table = table.reset_index()
table = ff.create_table(table)

salmon = '#FA8072'
scatter_plot = px.scatter(
    data_frame_of_data_of_General_Social_Survey,
    x = 'job_prestige',
    y = 'income',
    color = 'sex',
    color_discrete_map = {
        'male': 'blue',
        'female': salmon
    },
    trendline = 'ols',
    labels = {
        'job_prestige': 'occupational prestige', 
        'income': 'annual income'
    },
    hover_data = [
        'education',
        'socioeconomic_index'
    ]
)

data_frame_of_income_sex_and_job_prestige = data_frame_of_data_of_General_Social_Survey[['income', 'sex', 'job_prestige']]
data_frame_of_income_sex_and_job_prestige['job_prestige_binned'] = pd.cut(
    data_frame_of_income_sex_and_job_prestige.job_prestige,
    bins = 6
)
data_frame_of_income_sex_and_job_prestige = data_frame_of_income_sex_and_job_prestige.dropna()
data_frame_of_income_sex_and_job_prestige = data_frame_of_income_sex_and_job_prestige.sort_values(
    by = [
        'job_prestige_binned'
    ]
)
facet_grid = px.box(
    data_frame_of_income_sex_and_job_prestige,
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
facet_grid = facet_grid.for_each_annotation(
    lambda annotation: annotation.update(text = annotation.text.replace("job_prestige_binned=", "Occupational Prestige: "))
)

distributions_of_income_by_sex = px.box(
    data_frame_of_data_of_General_Social_Survey,
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
    data_frame_of_data_of_General_Social_Survey,
    x = 'sex',
    y = 'job_prestige',
    color = 'sex',
    color_discrete_map = {
        'male': 'blue', 'female': salmon
    },
    labels = {
        'job_prestige':'ocuupational prestige'
    },
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
    'Agreement With "A working mother can establish just as warm and secure a relationship with her children<br>as a mother who does not work."': 'relationship',
    'Agreement With "It is much better for everyone involved if the man is the achiever outside the home<br>and the woman takes care of the home and family."': 'male_breadwinner',
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