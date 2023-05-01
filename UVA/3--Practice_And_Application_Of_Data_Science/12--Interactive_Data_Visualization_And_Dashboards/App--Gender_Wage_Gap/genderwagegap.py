from callbacks import *
from dash import html, dcc
from datastructures import description_to_feature_whose_categories_forms_the_basis_of_bars, description_to_feature_by_which_to_group

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
                                    style = {
                                        'width': '100%'
                                    }
                                ),
                                html.Br(),
                                html.Button(
                                    'Socioeconomic Success By Sex',
                                    id = 'button_labeled_Socioeconomic_Success_By_Sex',
                                    style = {
                                        'width': '100%'
                                    }
                                ),
                                html.Br(),
                                html.Button(
                                    'Interactive Bar Plot',
                                    id = 'button_labeled_Interactive_Bar_Plot',
                                    style = {
                                        'width': '100%'
                                    }
                                ),
                                html.Br(),
                                html.Button(
                                    'Annual Income Versus Occupational Prestige',
                                    id = 'button_labeled_Annual_Income_Versus_Occupational_Prestige',
                                    style = {
                                        'width': '100%'
                                    }
                                ),
                                html.Br(),
                                html.Button(
                                    'Distributions Of Income By Sex And Occupational Prestige',
                                    id = 'button_labeled_Distributions_Of_Income_By_Sex_And_Occupational_Prestige',
                                    style = {
                                        'width': '100%'
                                    }
                                ),
                                html.Br(),
                                html.Button(
                                    'Distributions Of Income And Occupation Prestige By Sex',
                                    id = 'button_labeled_Distributions_Of_Income_And_Occupational_Prestige_By_Sex',
                                    style = {
                                        'width': '100%'
                                    }
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
                                # The following HTML division is a hidden copy of an HTML division returned by callback generate_figure.
                                # The presence of this HTML division is required so that the Python interpreter
                                # can associate this division's dropdowns with the inputs of callback generate_bar_plot and
                                # can associate this division's graph with the output of callback generate_bar_plot.
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
                                    style = {
                                        'display': 'none'
                                    }
                                )
                            ],
                            style = {
                                'border': 'solid'
                            }
                        )
                    ]
                )
            ],
            style = {
                'width': '100%'
            }
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug = True, port = 8051)