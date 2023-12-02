print("Enter between 1 and 5 student records.")
list_of_lists_of_student_information = []
should_continue = True
index = 0
while should_continue and index < 5:
    if index > 0:
        indicator_of_whether_loop_should_continue = input('should continue (y/n)? ')
        if indicator_of_whether_loop_should_continue == 'y':
            print('continuing')
        else:
            should_continue = False
            print('not continuing')
    index += 1
    if should_continue:
        print(f'Student {index}')
        ID = input('\tID: ')
        name = input('\tname: ')
        GPA = input('\tGPA: ')
        list_of_student_information = [ID, name, GPA]
        list_of_lists_of_student_information.append(list_of_student_information)
import pandas as pd
data_frame_of_student_information = pd.DataFrame(list_of_lists_of_student_information, columns = ['ID', 'name', 'GPA'])
print(data_frame_of_student_information)