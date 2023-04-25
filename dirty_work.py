import os 
import sys 

algo_list = os.listdir('logs')
for algo in algo_list:
    try:
        sub_dir = os.path.join('logs', algo)
        sub_list = os.listdir(sub_dir)

        for sub_path in sub_list:
            cur_path = os.path.join(sub_dir, sub_path)

            if '9999_wcup' in cur_path:
                cmd = 'mv ' + cur_path + ' ' + cur_path.replace('9999_wcup', '9999')
                # print(cmd)
                os.system(cmd)
    except Exception as e:
        pass