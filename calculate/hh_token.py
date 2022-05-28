# token.json
# {
# 	"client_id": "",
# 	"client_secret": "",
# 	"token_id": "",
# 	"token_type": ""
# }
import os.path


def get_token():
    full_path = os.path.abspath('') + '/data/token.json'
    token_dict = dict()
    token = ''
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf8') as file:
            exec('token_dict.update(' + file.read() + ')')
            file.close()
        hh_token = token_dict.get('token_id')
        hh_token_type = str.upper(token_dict.get('token_type')[0]) + token_dict.get('token_type')[1:]
        token_dict.update({'token_type': hh_token_type})
        token = hh_token_type + ' ' + hh_token
    else:
        print('Файл с данными для использования Токена не найден. Подложите файл (token.xml) с данными в папку "data"')

    return token

get_token()
