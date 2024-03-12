'''
joindatasets.py

Joins two datasets given:`
- a list of join keys as a string
- the name of an Excel worksheet with the first dataset
- the name of an Excel worksheet with the second dataset
- the number of rows to skip in the first dataset
- the number of rows to skip in the second dataset
- the path to the Excel workbook with the first dataset
- the path to the Excel workbook with the second dataset
- the type of join

Example usage:
python joindatasets.py '[EPA ID|Site Name]' Sites_Data Remedy_Data 13 0 Sites_Data.xlsx Remedy_Data.xlsx outer
'''

import argparse
import pandas as pd
    
def join_datasets(
    list_of_join_keys_for_dataset_1_as_string,
    list_of_join_keys_for_dataset_2_as_string,
    name_of_sheet_with_dataset_1,
    name_of_sheet_with_dataset_2,
    number_of_rows_to_skip_in_dataset_1,
    number_of_rows_to_skip_in_dataset_2,
    path_to_dataset_1,
    path_to_dataset_2,
    type_of_join
):
    if path_to_dataset_1.endswith('.xlsx') and path_to_dataset_2.endswith('.xlsx'):
        data_frame_1 = pd.read_excel(io = path_to_dataset_1, sheet_name = name_of_sheet_with_dataset_1, skiprows = number_of_rows_to_skip_in_dataset_1)
        data_frame_2 = pd.read_excel(io = path_to_dataset_2, sheet_name = name_of_sheet_with_dataset_2, skiprows = number_of_rows_to_skip_in_dataset_2)
        print('data frame 1:')
        print(data_frame_1)
        print('data_frame_2:')
        print(data_frame_2)
        list_of_join_keys_for_dataset_1 = list_of_join_keys_for_dataset_1_as_string.strip('[]').split('|')
        list_of_join_keys_for_dataset_2 = list_of_join_keys_for_dataset_2_as_string.strip('[]').split('|')
        column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_1 = data_frame_1[list_of_join_keys_for_dataset_1[0]]
        for i in range(1, len(list_of_join_keys_for_dataset_1)):
            column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_1 = column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_1 + '|' + data_frame_1[list_of_join_keys_for_dataset_1[i]]
        set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_1 = set(column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_1)
        column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_2 = data_frame_2[list_of_join_keys_for_dataset_2[0]]
        for i in range(1, len(list_of_join_keys_for_dataset_2)):
            column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_2 = column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_2 + '|' + data_frame_2[list_of_join_keys_for_dataset_2[i]]
        set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_2 = set(column_of_concatenated_elements_of_columns_corresponding_to_join_keys_of_data_frame_2)
        print(f'cardinality of set of concatenated elements corresponding to list of join keys of data frame 1: {len(set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_1)}')
        print(f'cardinality of set of concatenated elements corresponding to list of join keys of data frame 2: {len(set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_2)}')
        set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_1_but_not_data_frame_2 = set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_1 - set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_2
        set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_2_but_not_data_frame_1 = set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_2 - set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_1
        print(f'cardinality of set of concatenated elements corresponding to list of join keys of data frame 1 but not data frame 2: {len(set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_1_but_not_data_frame_2)}')
        print(f'cardinality of set of concatenated elements corresponding to list of join keys of data frame 2 but not data frame 1: {len(set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_2_but_not_data_frame_1)}')
        #data_frame_merged_on_EPA_ID = data_frame_1.merge(right = data_frame_2, how = type_of_join, left_on = ['EPA ID'], right_on = ['EPA ID'], suffixes = ('_1', '_2'))
        #data_frame_merged_on_EPA_ID.sort_values(by = ['EPA ID', 'Site Name_1', 'Site Name_2'], inplace = True)
        #print('data frame merged on EPA ID:')
        #print(data_frame_merged_on_EPA_ID)
        #data_frame_merged_on_Site_Name = data_frame_1.merge(right = data_frame_2, how = type_of_join, left_on = ['Site Name'], right_on = ['Site Name'], suffixes = ('_1', '_2'))
        #data_frame_merged_on_Site_Name.sort_values(by = ['EPA ID_1', 'EPA ID_2', 'Site Name'], inplace = True)
        #print('data frame merged on Site Name:')
        #print(data_frame_merged_on_Site_Name)
        data_frame_merged_on_list_of_join_keys = data_frame_1.merge(right = data_frame_2, how = type_of_join, left_on = list_of_join_keys_for_dataset_1, right_on = list_of_join_keys_for_dataset_2, sort = False, suffixes = ('_1', '_2'))
        data_frame_merged_on_list_of_join_keys.sort_values(by = list_of_join_keys_for_dataset_1, inplace = True)
        print('data frame merged on list of join keys:')
        print(data_frame_merged_on_list_of_join_keys)
        #for i in range(0, len(data_frame_merged_on_list_of_join_keys.index)):
        #    if (data_frame_merged_on_list_of_join_keys.at[i, 'Site Name'] != data_frame_merged_on_EPA_ID.at[i, 'Site Name_1']) and (data_frame_merged_on_list_of_join_keys.at[i, 'Site Name'] != data_frame_merged_on_EPA_ID.at[i, 'Site Name_2']):
        #        print(f"Site name {data_frame_merged_on_list_of_join_keys.at[i, 'Site Name']} of data frame merged on list of join keys has diverged from site name 1 {data_frame_merged_on_EPA_ID.at[i, 'Site Name_1']} and site name 2 {data_frame_merged_on_EPA_ID.at[i, 'Site Name_2']} of data frame merged on EPA ID, which suggests that site name 1 and site name 2 differ.")
        #        break
        column_of_concatenated_elements_of_columns_corresponding_to_list_of_join_keys_of_data_frame_merged_on_list_of_join_keys = data_frame_merged_on_list_of_join_keys[list_of_join_keys_for_dataset_1[0]]
        for i in range(1, len(list_of_join_keys_for_dataset_1)):
            column_of_concatenated_elements_of_columns_corresponding_to_list_of_join_keys_of_data_frame_merged_on_list_of_join_keys = column_of_concatenated_elements_of_columns_corresponding_to_list_of_join_keys_of_data_frame_merged_on_list_of_join_keys + '|' + data_frame_merged_on_list_of_join_keys[list_of_join_keys_for_dataset_1[i]]
        set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_merged_on_list_of_join_keys = set(column_of_concatenated_elements_of_columns_corresponding_to_list_of_join_keys_of_data_frame_merged_on_list_of_join_keys)
        print(f'cardinality of set of concatenated elements corresponding to list of join keys of data frame merged on list of join keys: {len(set_of_concatenated_elements_corresponding_to_list_of_join_keys_of_data_frame_merged_on_list_of_join_keys)}')
        list_of_join_keys_with_spaces_replaced_with_underscores = [join_key.replace(' ', '_') for join_key in list_of_join_keys_for_dataset_1]
        path_to_data_frame_merged_on_list_of_join_keys = 'Data_Frame_Merged_On_' + '_And_'.join(list_of_join_keys_with_spaces_replaced_with_underscores) + '.csv'
        data_frame_merged_on_list_of_join_keys.to_csv(path_or_buf = path_to_data_frame_merged_on_list_of_join_keys, index = False)
    else:
        raise NotImplementedError('joindatasets only supports loading datasets from Excel workbooks.')

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'Join Datasets', description = 'This program joins datasets.')
    parser.add_argument('list_of_join_keys_for_dataset_1_as_string', help = 'string of join keys for dataset 1 separated by pipes')
    parser.add_argument('list_of_join_keys_for_dataset_2_as_string', help = 'string of join keys for dataset 2 separated by pipes')
    parser.add_argument('name_of_sheet_with_dataset_1', help = 'name of sheet with dataset 1')
    parser.add_argument('name_of_sheet_with_dataset_2', help = 'name of sheet with dataset 2')
    parser.add_argument('number_of_rows_to_skip_in_dataset_1', help = 'number of rows to skip in dataset 1')
    parser.add_argument('number_of_rows_to_skip_in_dataset_2', help = 'number of rows to skip in dataset 2')
    parser.add_argument('path_to_dataset_1', help = 'path to dataset 1')
    parser.add_argument('path_to_dataset_2', help = 'path to dataset 2')
    parser.add_argument('type_of_join', help = 'type of join')
    args = parser.parse_args()
    list_of_join_keys_for_dataset_1_as_string = args.list_of_join_keys_for_dataset_1_as_string
    list_of_join_keys_for_dataset_2_as_string = args.list_of_join_keys_for_dataset_2_as_string
    name_of_sheet_with_dataset_1 = args.name_of_sheet_with_dataset_1
    name_of_sheet_with_dataset_2 = args.name_of_sheet_with_dataset_2
    number_of_rows_to_skip_in_dataset_1 = int(args.number_of_rows_to_skip_in_dataset_1)
    number_of_rows_to_skip_in_dataset_2 = int(args.number_of_rows_to_skip_in_dataset_2)
    path_to_dataset_1 = args.path_to_dataset_1
    path_to_dataset_2 = args.path_to_dataset_2
    type_of_join = args.type_of_join
    print(f'list of join keys for dataset 1 as string: {list_of_join_keys_for_dataset_1_as_string}')
    print(f'list of join keys for dataset 2 as string: {list_of_join_keys_for_dataset_2_as_string}')
    print(f'name of sheet with dataset 1: {name_of_sheet_with_dataset_1}')
    print(f'name of sheet with dataset 2: {name_of_sheet_with_dataset_2}')
    print(f'number of rows to skip in dataset 1: {number_of_rows_to_skip_in_dataset_1}')
    print(f'number of rows to skip in dataset 2: {number_of_rows_to_skip_in_dataset_2}')
    print(f'path to dataset 1: {path_to_dataset_1}')
    print(f'path to dataset 2: {path_to_dataset_1}')
    print(f'type of join: {type_of_join}')
    dictionary_of_arguments['list_of_join_keys_for_dataset_1_as_string'] = list_of_join_keys_for_dataset_1_as_string
    dictionary_of_arguments['list_of_join_keys_for_dataset_2_as_string'] = list_of_join_keys_for_dataset_2_as_string
    dictionary_of_arguments['name_of_sheet_with_dataset_1'] = name_of_sheet_with_dataset_1
    dictionary_of_arguments['name_of_sheet_with_dataset_2'] = name_of_sheet_with_dataset_2
    dictionary_of_arguments['number_of_rows_to_skip_in_dataset_1'] = number_of_rows_to_skip_in_dataset_1
    dictionary_of_arguments['number_of_rows_to_skip_in_dataset_2'] = number_of_rows_to_skip_in_dataset_2
    dictionary_of_arguments['path_to_dataset_1'] = path_to_dataset_1
    dictionary_of_arguments['path_to_dataset_2'] = path_to_dataset_2
    dictionary_of_arguments['type_of_join'] = type_of_join
    return dictionary_of_arguments

if __name__ == '__main__':
    dictionary_of_arguments = parse_arguments()
    join_datasets(
        list_of_join_keys_for_dataset_1_as_string = dictionary_of_arguments['list_of_join_keys_for_dataset_1_as_string'],
        list_of_join_keys_for_dataset_2_as_string = dictionary_of_arguments['list_of_join_keys_for_dataset_2_as_string'],
        name_of_sheet_with_dataset_1 = dictionary_of_arguments['name_of_sheet_with_dataset_1'],
        name_of_sheet_with_dataset_2 = dictionary_of_arguments['name_of_sheet_with_dataset_2'],
        number_of_rows_to_skip_in_dataset_1 = dictionary_of_arguments['number_of_rows_to_skip_in_dataset_1'],
        number_of_rows_to_skip_in_dataset_2 = dictionary_of_arguments['number_of_rows_to_skip_in_dataset_2'],
        path_to_dataset_1 = dictionary_of_arguments['path_to_dataset_1'],
        path_to_dataset_2 = dictionary_of_arguments['path_to_dataset_2'],
        type_of_join = dictionary_of_arguments['type_of_join']
    )