import pandas as pd
from os import getenv, environ
import bs4 as bs
import numpy as np

from nltk.tokenize import WordPunctTokenizer
from nltk.stem import snowball
from nltk.corpus import stopwords
from nltk import download
from pymorphy2 import MorphAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score as map2k

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from transformers import AutoTokenizer, AutoModel
import torch

from calculate.get_vacancies import hh_api_get_vacancies
from calculate.log_writer import to_log


def clean_tokenize(text=''):
    text = bs.BeautifulSoup(text, 'html.parser').get_text().lower()
    tokenizer = WordPunctTokenizer()
    ls_token = tokenizer.tokenize(text)
    rus_stem = snowball.SnowballStemmer(language='russian')
    morph = MorphAnalyzer()
    ls_new = []
    for word in ls_token:
        # приводим слово к нормальной форме
        morphed = morph.parse(word)[0].normal_form
        # stem_word = rus_stem.stem(morphed)
        # отбираем слова, которые несут смысловую нагрузку
        if morphed not in stopwords.words('russian') and morphed.isalpha():
            ls_new.append(rus_stem.stem(morphed))
    return ' '.join(ls_new)


def run_tfidf(cleared_user_text='', vacancies=pd.DataFrame(columns=['cleared_vacs'])):
    tfidf_vec = TfidfVectorizer(ngram_range=(2, 2), analyzer='word')
    # обучаемся
    tfidf_vec.fit([' '.join(vacancies['cleared_vacs'].to_list()) + ' ' + ' '.join(cleared_user_text)])
    # формируем векторы
    user_vec = tfidf_vec.transform([cleared_user_text])
    vacs_vec = tfidf_vec.transform(vacancies.cleared_vacs.to_list())
    # сравниваем векторы и получаем список схожести
    scores = cosine_similarity(user_vec, vacs_vec)[0]
    # ранжируем результаты сравнения
    tf_idf_top10 = pd.Series(scores).sort_values(ascending=False).index.to_list()[0:10]
    if int(getenv('razmetka_done')):
        y_true = pd.Series({'55088959': 0, '55486773': 0, '55318384': 1, '66109317': 1, '55514389': 1, '55302174': 1,
                            '55306481': 1, '66107360': 1, '54300274': 0, '66109316': 0, '55319016': 0, '66109318': 0,
                            '54825318': 0, '55517668': 0, '55302220': 0, '54552531': 1, '55302191': 0, '55594648': 1,
                            '50630870': 1, '52396406': 1, '54692864': 1, '55458328': 1, '55588931': 0, '66146489': 0,
                            '54717873': 0, '54659098': 1, '54800651': 1}, name='fact')
        ids = pd.Series(scores).sort_values(ascending=False).index.to_list()[:10]
        scores = list(scores)
        scores.sort(reverse=True)
        scores = scores[:10]
        res_map = pd.Series(scores, index=vacancies.iloc[ids].index.to_list(), name='score')
        df_map = pd.concat([res_map, y_true], axis=1, join='inner')
        arr_score = np.array(df_map['score'].to_list())
        arr_true = np.array(df_map['fact'].to_list())
        m2k = map2k(arr_true, arr_score)
        to_log(log_text=f'Метрика map@k для модели TF-IDF: {m2k}')
    to_log(log_text='TF-IDF: завершили')

    return vacancies['url'].iloc[tf_idf_top10].to_list()


def run_word2vec(cleared_user_text='', vacancies=pd.DataFrame(columns=['cleared_vacs'])):
    # задаем параметры для модели
    max_epochs = 30
    vec_size = 125
    alpha = 0.025
    data = [cleared_user_text] + vacancies['cleared_vacs'].to_list()
    # todo: добавить комментарии по модели word2vec
    tagged_data = [TaggedDocument(words=sentence.split(' '), tags=[str(i)]) for i, sentence in enumerate(data)]
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    # todo: добавить комментарии по модели word2vec
    model.build_vocab(tagged_data)
    # todo: добавить комментарии по модели word2vec
    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs + 1)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    # токенизируем сообщение пользователя
    user_data = cleared_user_text.split(' ')
    # формируем вектор пользовательского описания
    user_vec = model.infer_vector(user_data)
    # формируем векторы описания вакансий
    vac_vecs = [model.infer_vector(x.split(' ')) for x in vacancies['cleared_vacs'].to_list()]
    # сравниваем векторы и получаем список схожести
    scores = model.dv.cosine_similarities(user_vec, vac_vecs)
    # ранжируем результаты сравнения
    res_top_list = pd.Series(scores).sort_values(ascending=False).index.to_list()[:10]
    if int(getenv('razmetka_done')):
        y_true = pd.Series({'55088959': 0, '55486773': 0, '55318384': 1, '66109317': 1, '55514389': 1, '55302174': 1,
                            '55306481': 1, '66107360': 1, '54300274': 0, '66109316': 0, '55319016': 0, '66109318': 0,
                            '54825318': 0, '55517668': 0, '55302220': 0, '54552531': 1, '55302191': 0, '55594648': 1,
                            '50630870': 1, '52396406': 1, '54692864': 1, '55458328': 1, '55588931': 0, '66146489': 0,
                            '54717873': 0, '54659098': 1, '54800651': 1}, name='fact')
        ids = pd.Series(scores).sort_values(ascending=False).index.to_list()[:10]
        scores = list(scores)
        scores.sort(reverse=True)
        scores = scores[:10]
        res_map = pd.Series(scores, index=vacancies.iloc[ids].index.to_list(), name='score')
        df_map = pd.concat([res_map, y_true], axis=1, join='inner')
        arr_score = np.array(df_map['score'].to_list())
        arr_true = np.array(df_map['fact'].to_list())
        m2k = map2k(arr_true, arr_score)
        to_log(log_text=f'Метрика map@k для модели Word2Vec: {m2k}')
    to_log(log_text='Word2Vec: завершили')

    return vacancies['url'].iloc[res_top_list].to_list()


def run_sbert(cleared_user_text='', vacancies=pd.DataFrame(columns=['cleared_vacs'])):
    # todo: добавить комментарии по модели sbert
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    vacs_text = vacancies['cleared_vacs'].to_list()
    user_text = [cleared_user_text]
    # all_texts = [cleared_user_text] + vacancies['cleared_vacs'].to_list()
    scores = []
    for iter_text in vacs_text:
        all_texts = user_text + [iter_text]
        # todo: добавить комментарии по модели sbert
        tokens = tokenizer(all_texts,
                           max_length=128,
                           truncation=True,
                           padding='max_length',
                           return_tensors='pt')
        # todo: добавить комментарии по модели sbert
        outputs = model(**tokens)
        # todo: добавить комментарии по модели sbert
        embeddings = outputs.last_hidden_state
        # todo: добавить комментарии по модели sbert
        mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        # todo: добавить комментарии по модели sbert
        masked_embeddings = embeddings * mask
        # todo: добавить комментарии по модели sbert
        # todo: добавить комментарии по модели sbert
        summed = torch.sum(masked_embeddings, 1)
        # todo: добавить комментарии по модели sbert
        counted = torch.clamp(mask.sum(1), min=1e-9)
        # todo: добавить комментарии по модели sbert
        mean_pooled = summed / counted
        # todo: добавить комментарии по модели sbert
        mean_pooled = mean_pooled.detach().numpy()
        # todo: добавить комментарии по модели sbert
        # scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
        # scores = np.zeros((1, mean_pooled.shape[0]-1))
        # todo: добавить комментарии по модели sbert
        score = cosine_similarity([mean_pooled[0]], mean_pooled[1:])
        # получаем список схожести
        scores.append(score[0])
    list_res = pd.Series(scores).sort_values(ascending=False).index.to_list()[:10]
    if int(getenv('razmetka_done')):
        y_true = pd.Series({'55088959': 0, '55486773': 0, '55318384': 1, '66109317': 1, '55514389': 1, '55302174': 1,
                            '55306481': 1, '66107360': 1, '54300274': 0, '66109316': 0, '55319016': 0, '66109318': 0,
                            '54825318': 0, '55517668': 0, '55302220': 0, '54552531': 1, '55302191': 0, '55594648': 1,
                            '50630870': 1, '52396406': 1, '54692864': 1, '55458328': 1, '55588931': 0, '66146489': 0,
                            '54717873': 0, '54659098': 1, '54800651': 1}, name='fact')
        ids = pd.Series(scores).sort_values(ascending=False).index.to_list()[:10]
        scores = list(scores)
        scores.sort(reverse=True)
        scores = scores[:10]
        res_map = pd.Series(scores, index=vacancies.iloc[ids].index.to_list(), name='score')
        df_map = pd.concat([res_map, y_true], axis=1, join='inner')
        arr_score = np.array(df_map['score'].to_list())
        arr_true = np.array(df_map['fact'].to_list())
        m2k = map2k(arr_true, arr_score)
        to_log(log_text=f'Метрика map@k для модели Sentence-BERT: {m2k}')
    to_log(log_text='S-BERT: завершили')

    return vacancies['url'].iloc[list_res].to_list()


def nlp_predict(user_text='', vacancies=pd.DataFrame()):
    # предобработка текста
    to_log(log_text='Запускаем предобработку текста скаченных вакансий и текста от пользователя')
    vacancies['cleared_vacs'] = vacancies['description'].apply(lambda x: clean_tokenize(x))
    cleared_user_text = clean_tokenize(text=user_text)
    to_log(log_text='Закончили предобработку текста скаченных вакансий')

    # рабочая модель (для разметки используем весь набор)
    mode_rezhim = int(getenv('model'))
    if mode_rezhim == 0:
        to_log(log_text='TF-IDF: запускаем обработку')
        tfidf = run_tfidf(cleared_user_text, vacancies)
        to_log(log_text='Word2Vec: запускаем обработку')
        word2vec = run_word2vec(cleared_user_text, vacancies)
        to_log(log_text='S-BERT: запускаем обработку')
        run_bert = run_sbert(cleared_user_text, vacancies)

        return tfidf + word2vec + run_bert
    elif mode_rezhim == 1:
        to_log(log_text='TF-IDF: запускаем обработку')
        return run_tfidf(cleared_user_text, vacancies)
    elif mode_rezhim == 2:
        to_log(log_text='Word2Vec: запускаем обработку')
        return run_word2vec(cleared_user_text, vacancies)
    elif mode_rezhim == 3:
        to_log(log_text='S-BERT: запускаем обработку')
        return run_sbert(cleared_user_text, vacancies)
    else:
        environ['model'] = '1'
        return ['Проблема с настройками. Используется модель по умолчанию: TF-IDF'] + run_tfidf(cleared_user_text,
                                                                                                vacancies)


if __name__ == '__main__':
    # данные для разметки
    environ['razmetka_done'] = '1'
    download('stopwords')
    user_text = 'хочу быть data scientist, окончил школу data analytic и школу data scientist корпоративного ' \
                'университета сбербанка. очень хорошо знаю sql от написания сложных запросов с использование ' \
                'оконных функций, регулярных выражений, иерархических запросов, до pl/sql скриптов. Есть профиль ' \
                'с работами на github, работал со всем необходимыми библиотеками: pandas, numpy, seaborn, plotly, ' \
                'mathplotlib, sklearn, есть базовые работы с deep learning на MNIST, есть опыт в работе с ' \
                'библиотеками для NLP: TF-IDF, word2vec, BERT, SBERT. Имею огромное желание развиваться ' \
                'направлении DA и DS'
    df_vacs = hh_api_get_vacancies(user_id=int(getenv('bot_admin')))
    res = nlp_predict(user_text=user_text, vacancies=df_vacs)
    to_log(log_text=f"Результат подбора:\r\n{res}")

