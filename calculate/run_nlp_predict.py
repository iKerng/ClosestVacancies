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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from transformers import AutoTokenizer, AutoModel
import torch

from calculate.get_vacancies import hh_api_get_vacancies


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
    result_pair = cosine_similarity(user_vec, vacs_vec)[0]
    # ранжируем результаты сравнения
    tf_idf_top10 = pd.Series(result_pair).sort_values(ascending=False).index.to_list()[0:10]
    print('TF-IDF: завершили')

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
    res = model.dv.cosine_similarities(user_vec, vac_vecs)
    # ранжируем результаты сравнения
    res_top_list = pd.Series(res).sort_values(ascending=False).index.to_list()[:10]
    print('Word2Vec: завершили')

    return vacancies['url'].iloc[res_top_list]


def run_sbert(cleared_user_text='', vacancies=pd.DataFrame(columns=['cleared_vacs'])):
    # todo: добавить комментарии по модели sbert
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    all_texts = [cleared_user_text] + vacancies['cleared_vacs'].to_list()
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
    summed = torch.sum(masked_embeddings, 1)
    # todo: добавить комментарии по модели sbert
    counted = torch.clamp(mask.sum(1), min=1e-9)
    # todo: добавить комментарии по модели sbert
    mean_pooled = summed / counted
    # todo: добавить комментарии по модели sbert
    mean_pooled = mean_pooled.detach().numpy()
    # todo: добавить комментарии по модели sbert
    scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
    # todo: добавить комментарии по модели sbert
    scores[0, :] = cosine_similarity([mean_pooled[0]], mean_pooled)[0]
    # получаем список схожести
    res = list((scores[0])[1:])
    # ранжируем результаты сравнения
    list_res = pd.Series(res).sort_values(ascending=False).index.to_list()[:10]
    print('S-BERT: завершили')

    return vacancies['url'].iloc[list_res]


def nlp_predict(user_text='', vacancies=pd.DataFrame()):
    # предобработка текста
    vacancies['cleared_vacs'] = vacancies['description'].apply(lambda x: clean_tokenize(x))
    cleared_user_text = clean_tokenize(text=user_text)

    # рабочая модель (для разметки используем весь набор)
    mode_rezhim = int(getenv('model'))
    if mode_rezhim == 0:
        print('TF-IDF: запускаем обработку')
        tfidf = run_tfidf(cleared_user_text, vacancies)
        print('Word2Vec: запускаем обработку')
        word2vec = run_word2vec(cleared_user_text, vacancies)
        print('S-BERT: запускаем обработку')
        run_sber = run_sbert(cleared_user_text, vacancies)
        return tfidf + word2vec + run_sber
    elif mode_rezhim == 1:
        print('TF-IDF: запускаем обработку')
        return run_tfidf(cleared_user_text, vacancies)
    elif mode_rezhim == 2:
        print('Word2Vec: запускаем обработку')
        return run_word2vec(cleared_user_text, vacancies)
    elif mode_rezhim == 3:
        print('S-BERT: запускаем обработку')
        return run_sbert(cleared_user_text, vacancies)
    else:
        environ['model'] = '1'
        return ['Проблема с настройками. Используется модель по умолчанию: TF-IDF'] + run_tfidf(cleared_user_text,
                                                                                                vacancies)


if __name__ == '__main__':
    # данные для разметки
    download('stopwords')
    user_text = 'хочу быть data scientist, окончил школу data analytic и школу data scientist корпоративного ' \
                'университета сбербанка. очень хорошо знаю sql от написания сложных запросов с использование ' \
                'оконных функций, регулярных выражений, иерархических запросов, до pl/sql скриптов. Есть профиль ' \
                'с работами на github, работал со всем необходимыми библиотеками: pandas, numpy, seaborn, plotly, ' \
                'mathplotlib, sklearn, есть базовые работы с deep learning на MNIST, есть опыт в работе с ' \
                'библиотеками для NLP: TF-IDF, word2vec, BERT, SBERT. Имею огромное желание развиваться ' \
                'направлении DA и DS'
    df_vacs = hh_api_get_vacancies(user_id=int(getenv('bot_admin')))
    tfidf, w2v, sbert = nlp_predict(user_text=user_text, vacancies=df_vacs)
    tfidf = '\r\n'.join(tfidf)
    w2v = '\r\n'.join(w2v)
    sbert = '\r\n'.join(sbert)
    print(f"Результат подбора по версии TF-IDF:\r\n{tfidf}")
    print(f"Результат подбора по версии Word2Vec:\r\n{tfidf}")
    print(f"Результат подбора по версии BERT:\r\n{tfidf}")
    # todo: сделать разметку вакансий
    #  сделать сравнение предсказаний моделей с помощью map2k
