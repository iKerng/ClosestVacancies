from os import path, getcwd
import datetime as dt

def to_log(log_user='Application', log_user_id=0, log_text=''):
    path_to_log_file = path.abspath(getcwd() + '/logs/' + dt.datetime.utcnow().strftime('%Y-%m-%d') + '.log')
    print(path_to_log_file)
    cur_time = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
    log_text = log_text.replace('\r\n','<new_line>').replace('\r','<new_line>').replace('\n','<new_line>')
    with open(path_to_log_file, 'a', encoding='utf8') as file:
        log_text = f"{cur_time}: [{'@' + str(log_user) if log_user != 'Application' else 'Application'}, " \
                   f"{str(log_user_id)}]: {log_text}.\r\n"
        file.write(log_text)
        file.close()
