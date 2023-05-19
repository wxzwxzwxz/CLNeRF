import numpy as np

def print_line_for_table1(name, result, our_algo = False):
    '''
    Print line for table1
    :param name: the algorithm name
    :param result: the result of the algorithm
            tamplate = [Add_Old, Add_new, Delete_Old, Delete_New, Move_Old, Move_New, Replace_Old, Replace_New]
    :param our_algo: judge whether is our algorithm
    :return:
    '''
    if our_algo == True:
        table_line = r'\rowcolor{mygray}\textbf{'+name+'}'
        for result_i in result:
            table_line = table_line +'&'+  r'\textbf{%.3f}' % (float(result_i)) # + str(result_i) + '}'
        table_line = table_line + r'\\'
    else:
        table_line = r'\textbf{'+name+'}'
        for result_i in result:
            table_line = table_line +'& %.3f ' % (float(result_i)) # +  str(result_i)
        table_line = table_line + r'\\'
    return table_line

def latex_for_table1(results, our_algo_name):
    '''
    output whole latex for table1
    :param results: results_dict includes all algorithms
    :param our_algo_name: use to judge if our_algorithm to generate textbf and rowcolor in table
    :return: nothing
    '''

    for key in results:
        if key == our_algo_name:
            our_algo = True
            print(print_line_for_table1(key, results[key], our_algo))
        else:
            our_algo = False
            print(print_line_for_table1(key, results[key], our_algo))
    return

'''
results = np.zeros(4,4)
'''
results_table2 = dict()
results = np.zeros((4,4))
results_table2[r'algorihtm_name'] = results
print(results_table2[r'algorihtm_name'][1,1])
# def print_line_for_table2(result, our_algo = False, table_row_1, table_row_2, table_row_3, table_row_4):
#     '''
#     Print line for table2
#     :param result: result matrix
#     :param our_algo: is our algorithm or not
#     :param table_row_1:
#     :param table_row_2:
#     :param table_row_3:
#     :param table_row_4:
#     :return:
#     '''
#     if our_algo == True:
#         table_line = r'\rowcolor{mygray}\textbf{'+name+'}'
#         for result_i in result:
#             table_line = table_line +'&'+  r'\textbf{' + str(result_i) + '}'
#         table_line = table_line + r'\\'
#     else:
#         table_line = r'\textbf{'+name+'}'
#         for result_i in result:
#             table_line = table_line +'&'+  str(result_i)
#         table_line = table_line + r'\\'
#     return table_line

def latex_for_table2(results, our_algo_name):
    '''
    output whole latex for table2
    :param results: results_dict includes all algorithms each result is a matrix
    :param our_algo_name: use to judge if our_algorithm to generate textbf and rowcolor in table
    :return: nothing
    '''
    table_row = list()
    table_row1 = r'\multicolumn{1}{c|}{\multirow{4}{*}{\rotatebox{90}{\textbf{Training on}}}}& \textit{Task1}'
    table_row2 = r'\multicolumn{1}{c|}{} & \textit{Task2}'
    table_row3 = r'\multicolumn{1}{c|}{} & \textit{Task3}'
    table_row4 = r'\multicolumn{1}{c|}{} & \textit{Task4}'
    table_row.append(table_row1)
    table_row.append(table_row2)
    table_row.append(table_row3)
    table_row.append(table_row4)
    for key in results: # algorithm
        if key == our_algo_name: # format
            for task_index in range(len(table_row)): # rowline
                for value in results[key][task_index]:
                    # table_row[task_index] = table_row[task_index] + r'&\textbf{' +str(value) + '}'
                    if value < 0:
                        table_row[task_index] = table_row[task_index] + r'& - '
                    else:
                        table_row[task_index] = table_row[task_index] + r'&\textbf{%.3f}' % (value)
        else:
            for task_index in range(len(table_row)):  # rowline
                for value in results[key][task_index]:
                    if value < 0:
                        table_row[task_index] = table_row[task_index] + r'& - '
                    else:
                        # table_row[task_index] = table_row[task_index] + r'&' + str(value)
                        table_row[task_index] = table_row[task_index] + r'& {%.3f}' % value
                    
    for result_line in table_row:
        print(result_line + r'\\')
    return

if __name__ == '__main__':
    '''
    Tamplate of input for table1
    results_table1 = dict()
    results_table1['algorithm_name'] = [0.0(Add_Old), 0.0(Add_New), 0.0(Delete_Old), 0.0(Delete_New), 0.0(Move_Old), 0.0(Move_New), 0.0(Replace_Old), 0.0(Replace_New)]
    '''
    results_table1 = dict()
    	
	# A
    results_table1['FT'] = [24.212, 23.540, 24.138, 34.758, 20.931, 29.285, 21.453, 30.143, 16.915, 27.362]
    results_table1['MR'] = [31.967, 19.403, 30.024, 17.206, 32.388, 25.656, 32.476, 25.154, 25.946, 24.450]
    results_table1['DyNeRF'] = [32.291, 23.890, 31.395, 25.823, 32.832, 30.124, 32.700, 29.876, 0.000, 0.000]
    results_table1['Ours'] = [32.328, 25.293, 32.391, 34.773, 33.112, 29.814, 33.221, 29.640, 31.699, 29.054]
	
    # A
    # results_table1['KD'] = [31.243, 23.417, 29.924, 27.478,	
    # results_table1['Expert'] = [31.788, 24.096, 29.871, 34.574, 
    # results_table1['Ours'] = [32.328, 25.293, 32.391, 34.773, 33.347, 29.626, 33.221, 29.640, 31.699, 29.054]


    # B
    # results_table1['FT'] = [26.831, 25.475, 31.059, 23.707, 23.024, 23.334, 28.966, 25.131, 0.000, 0.000]
    # results_table1['MR'] = [25.942, 28.995, 31.045, 29.001, 21.871, 28.851, 29.681, 28.971, 0.000, 0.000]
    # results_table1['DyNeRF'] = [27.514, 29.251, 23.389, 29.083, 23.150, 28.838, 23.251, 29.161, 0.000, 0.000]
    # results_table1['Ours'] = [27.023, 27.885, 30.653, 28.809, 0.000, 0.000, 29.693, 28.425, 0.000, 0.000]
    latex_for_table1(results_table1, 'Ours')

    print()
    print()
    
    '''
    Tamplate of input for table2
    results_table2 = dict()
    results_table2['algorithm_name'] = 
    [
    [1,1  1,2  1,3  1,4]
    [2,1  2,2  2,3  2,4]
    [3,1  3,2  3,3  3,4]
    [4,1  4,2  4,3  4,4]
    ]
    train_task_id, test_task_id
    '''
    
    results_table2 = dict()
    # FT
    results_table2['FT'] = np.zeros((4,4))
    results_table2['FT'][0] = [-1, -1, -1, -1]
    results_table2['FT'][1] = [20.64543646, -1, -1, -1]
    results_table2['FT'][2] = [16.81103018, 21.54880912, -1, -1]
    results_table2['FT'][3] = [14.63479144, 20.2762135, 18.58333889, -1]
    results_table2['FT'][0][0] = results_table1['FT'][1]
    results_table2['FT'][1][1] = results_table1['FT'][3]
    results_table2['FT'][2][2] = results_table1['FT'][5]
    results_table2['FT'][3][3] = results_table1['FT'][7]
    
    # MR
    results_table2['MR'] = -1 * np.ones((4,4))
    results_table2['MR'][0] = [-1, -1, -1, -1]
    results_table2['MR'][1] = [17.48201167, -1, -1, -1]
    results_table2['MR'][2] = [17.46424644, 17.63404493, -1, -1]
    results_table2['MR'][3] = [17.49236171, 17.8028726, 23.27553668, -1]
    results_table2['MR'][0][0] = results_table1['MR'][1]
    results_table2['MR'][1][1] = results_table1['MR'][3]
    results_table2['MR'][2][2] = results_table1['MR'][5]
    results_table2['MR'][3][3] = results_table1['MR'][7]

    # Ours
    results_table2['Ours'] = -1 * np.ones((4,4))
    results_table2['Ours'][0] = [-1, -1, -1, -1]
    results_table2['Ours'][1] = [24.40780539, -1, -1, -1]
    results_table2['Ours'][2] = [24.05414591, 33.77792289, -1, -1]
    results_table2['Ours'][3] = [23.71442592, 33.07247268, 24.94351561, -1]
    results_table2['Ours'][0][0] = results_table1['Ours'][1]
    results_table2['Ours'][1][1] = results_table1['Ours'][3]
    results_table2['Ours'][2][2] = results_table1['Ours'][5]
    results_table2['Ours'][3][3] = results_table1['Ours'][7]

    latex_for_table2(results_table2, 'Ours')
