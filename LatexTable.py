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
            table_line = table_line +'&'+  r'\textbf{%.2f}' % (float(result_i)) # + str(result_i) + '}'
        table_line = table_line + r'\\'
    else:
        table_line = r'\textbf{'+name+'}'
        for result_i in result:
            table_line = table_line +'& %.2f ' % (float(result_i)) # +  str(result_i)
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
                        table_row[task_index] = table_row[task_index] + r'&\textbf{%.2f}' % (value)
        else:
            for task_index in range(len(table_row)):  # rowline
                for value in results[key][task_index]:
                    if value < 0:
                        table_row[task_index] = table_row[task_index] + r'& - '
                    else:
                        # table_row[task_index] = table_row[task_index] + r'&' + str(value)
                        table_row[task_index] = table_row[task_index] + r'& {%.2f}' % value
                    
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
    # results_table1['FT'] = [24.212, 23.540, 22.895, 34.758, 20.931, 29.285, 21.453, 30.143, 16.915, 27.362]
    # results_table1['MR'] = [31.967, 19.403, 31.401, 17.206, 32.388, 25.656, 32.476, 25.154, 25.946, 24.450]
    # results_table1['DyNeRF'] = [32.291, 23.890, 31.395, 25.823, 32.832, 30.124, 32.700, 29.876, 25.805, 29.831]
    # results_table1['Ours'] = [32.328, 25.293, 32.429, 34.773, 33.297, 29.970, 33.332, 29.816, 33.376, 29.739]

    # # A	
    # results_table1['w/o Expert'] = [31.243, 23.417, 30.358, 27.478, 31.231, 30.857, 31.327, 30.002, 30.385, 24.634]
    # results_table1['w/o Expert'] = [31.243, 23.417, 30.358, 27.478, 31.231, 30.857, 31.327, 30.002, 29.385, 24.297]
    # results_table1['w/o KD'] = [29.812, 24.096, 29.273, 34.574, 23.903, 29.924, 23.468, 28.974, 18.629, 28.194]
    # results_table1['w/o $L_{m}$'] = [-1, -1, -1, -1, -1, -1, -1, -1, 33.376, 29.739]
    # results_table1['w/o $L_{m}$'] = [-1, -1, -1, -1, -1, -1, -1, -1, 33.376, 29.739]
    # results_table1['w/o $L_{m}$'] = [31.084, 24.918, 30.986, 31.735, 31.898, 28.637, 32.627, 28.534, 30.699, 28.054]
    # results_table1['Ours'] = [32.328, 25.293, 32.429, 34.773, 33.297, 29.970, 33.332, 29.816, 33.376, 29.739]

    # B
    results_table1['FT'] = [25.475, 26.831, 23.707, 31.059, 23.334, 23.024, 25.131, 28.966, 23.061, 28.577]
    results_table1['MR'] = [28.995, 25.942, 28.107, 26.881, 28.588, 19.971, 28.953, 29.171, 27.627, 29.412]
    results_table1['DyNeRF'] = [29.170, 27.456, 29.083, 23.389, 28.599, 22.164, 29.161, 23.251, 27.625, 30.112]
    results_table1['Ours'] = [29.317, 27.435, 29.051, 31.091, 28.613, 23.090, 29.332, 30.163, 28.057, 30.071]
    latex_for_table1(results_table1, 'Ours')

    # A wo old
    # 27.399, 23.541, 27.648, 34.684, 27.119, 31.215, 26.001, 30.010	
    # B wo old
    # 27.334 & 27.188 & 27.619 & 31.276 & 25.479 & 22.127 & 27.745 & 30.204
    # 27.334 & 27.188 & 27.619 & 31.276 & 25.479 & 22.127 & 27.692 & 30.104

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

    # A
    # # FT
    # results_table2['FT'] = np.zeros((4,4))
    # results_table2['FT'][0] = [-1, -1, -1, -1]
    # results_table2['FT'][1] = [20.64543646, -1, -1, -1]
    # results_table2['FT'][2] = [16.81103018, 21.54880912, -1, -1]
    # results_table2['FT'][3] = [14.63479144, 20.2762135, 18.58333889, -1]
    # results_table2['FT'][0][0] = results_table1['FT'][1]
    # results_table2['FT'][1][1] = results_table1['FT'][3]
    # results_table2['FT'][2][2] = results_table1['FT'][5]
    # results_table2['FT'][3][3] = results_table1['FT'][7]
    
    # # MR
    # results_table2['MR'] = -1 * np.ones((4,4))
    # results_table2['MR'][0] = [-1, -1, -1, -1]
    # results_table2['MR'][1] = [17.48201167, -1, -1, -1]
    # results_table2['MR'][2] = [17.46424644, 17.63404493, -1, -1]
    # results_table2['MR'][3] = [17.49236171, 17.8028726, 23.27553668, -1]
    # results_table2['MR'][0][0] = results_table1['MR'][1]
    # results_table2['MR'][1][1] = results_table1['MR'][3]
    # results_table2['MR'][2][2] = results_table1['MR'][5]
    # results_table2['MR'][3][3] = results_table1['MR'][7]

    # # # DyNeRF
    # # results_table2['DyNeRF'] = -1 * np.ones((4,4))
    # # results_table2['DyNeRF'][0] = [-1, -1, -1, -1]
    # # results_table2['DyNeRF'][1] = [20.92407136, -1, -1, -1]
    # # results_table2['DyNeRF'][2] = [26.1020561, 19.02506679, -1, -1]
    # # results_table2['DyNeRF'][3] = [26.16828318, 19.79502276, 30.59813558, -1]
    # # results_table2['DyNeRF'][0][0] = results_table1['DyNeRF'][1]
    # # results_table2['DyNeRF'][1][1] = results_table1['DyNeRF'][3]
    # # results_table2['DyNeRF'][2][2] = results_table1['DyNeRF'][5]
    # # results_table2['DyNeRF'][3][3] = results_table1['DyNeRF'][7]

    # # Ours
    # results_table2['Ours'] = -1 * np.ones((4,4))
    # results_table2['Ours'][0] = [-1, -1, -1, -1]
    # results_table2['Ours'][1] = [24.40780539, -1, -1, -1]
    # results_table2['Ours'][2] = [24.05414591, 33.77792289, -1, -1]
    # # results_table2['Ours'][3] = [23.71442592, 33.07247268, 24.94351561, -1]
    # results_table2['Ours'][3] = [23.71442592, 33.07247268, 28.12931519, -1]
    # results_table2['Ours'][0][0] = results_table1['Ours'][1]
    # results_table2['Ours'][1][1] = results_table1['Ours'][3]
    # results_table2['Ours'][2][2] = results_table1['Ours'][5]
    # results_table2['Ours'][3][3] = results_table1['Ours'][7]

    # B
    # FT
    results_table2['FT'] = np.zeros((4,4))
    
    results_table2['FT'][0] = [-1, -1, -1, -1]
    results_table2['FT'][1] = [23.17809806, -1, -1, -1]
    results_table2['FT'][2] = [19.54439688, 25.32269421, -1, -1]
    results_table2['FT'][3] = [18.35521144, 24.32507094, 18.8268163, -1]
    results_table2['FT'][0][0] = results_table1['FT'][1]
    results_table2['FT'][1][1] = results_table1['FT'][3]
    results_table2['FT'][2][2] = results_table1['FT'][5]
    results_table2['FT'][3][3] = results_table1['FT'][7]
    
    # MR
    results_table2['MR'] = -1 * np.ones((4,4))
    results_table2['MR'][0] = [-1, -1, -1, -1]
    results_table2['MR'][1] = [23.69385879, -1, -1, -1]
    results_table2['MR'][2] = [23.71472661, 26.97677445, -1, -1]
    results_table2['MR'][3] = [23.59763464, 26.92283348, 18.83363197, -1]
    results_table2['MR'][0][0] = results_table1['MR'][1]
    results_table2['MR'][1][1] = results_table1['MR'][3]
    results_table2['MR'][2][2] = results_table1['MR'][5]
    results_table2['MR'][3][3] = results_table1['MR'][7]

    # # DyNeRF
    # results_table2['DyNeRF'] = -1 * np.ones((4,4))
    # results_table2['DyNeRF'][0] = [-1, -1, -1, -1]
    # results_table2['DyNeRF'][1] = [-1, -1, -1, -1]
    # results_table2['DyNeRF'][2] = [-1, -1, -1, -1]
    # results_table2['DyNeRF'][3] = [-1, -1, -1, -1]
    # results_table2['DyNeRF'][0][0] = results_table1['DyNeRF'][1]
    # results_table2['DyNeRF'][1][1] = results_table1['DyNeRF'][3]
    # results_table2['DyNeRF'][2][2] = results_table1['DyNeRF'][5]
    # results_table2['DyNeRF'][3][3] = results_table1['DyNeRF'][7]

    # Ours
    results_table2['Ours'] = -1 * np.ones((4,4))
    results_table2['Ours'][0] = [-1, -1, -1, -1]
    results_table2['Ours'][1] = [26.77180272, -1, -1, -1]
    results_table2['Ours'][2] = [26.25006047, 28.13322532, -1, -1]
    results_table2['Ours'][3] = [25.9655335, 27.92497623, 22.2881861, -1]
    results_table2['Ours'][0][0] = results_table1['Ours'][1]
    results_table2['Ours'][1][1] = results_table1['Ours'][3]
    results_table2['Ours'][2][2] = results_table1['Ours'][5]
    results_table2['Ours'][3][3] = results_table1['Ours'][7]

    latex_for_table2(results_table2, 'Ours')
