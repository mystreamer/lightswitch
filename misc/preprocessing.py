#!/usr/bin/env python3
from os import read
#from parse_html import standardize_dates
import spacy
import pandas as pd
import numpy as np
import re
import nltk
from collections import Counter
# from langdetect import detect
from nltk.corpus import stopwords
#import ssl
# import dateparser
import logging


logging.basicConfig(filename='preprocessing.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
'''
nltk.download('stopwords')
# python3 -m spacy download de_core_news_sm


def remove_newline_chars(text_list):
    articles = [re.sub('\s+', ' ', item) for item in text_list] #remove newline characters

    return articles

def to_lowercase(text_list):
    articles = [item.lower() for item in text_list]

    return articles


def preprocess(text_list, language_list, pos_tags_to_keep, filter_list):
    nlp_en = spacy.load("en_core_web_sm")
    stop_words_en = set(stopwords.words('english')) # nltk stopword list

    nlp_de = spacy.load('de_core_news_sm')
    stop_words_de = set(stopwords.words('german'))

    nlp_fr = spacy.load('fr_core_news_sm')
    stop_words_fr = set(stopwords.words('french'))

    nlp_it = spacy.load('it_core_news_sm')
    stop_words_it = set(stopwords.words('italian'))

    lang_modul = {'en': nlp_en, 'de': nlp_de, 'fr': nlp_fr, 'it': nlp_it}
    stop_lang = {'en': stop_words_en, 'de': stop_words_de, 'fr': stop_words_fr, 'it': stop_words_it}
    counter = 1
    lemmatized_text = []
    for article, lang_id in zip(text_list, language_list):
        tokenized_article = []
        try:        
            stop_w_list = stop_lang[lang_id]
        except KeyError: # default german
            stop_w_list = stop_lang['de']
        try:
            doc = lang_modul[lang_id](article)
        except KeyError:
            doc = lang_modul['de'](article)
            
        for token in doc:
            if token.text.find('\n') != 0: # remove newline characters
                if token.text.lower() not in stop_w_list: # remove stopwords
                    if token.text.lower() not in filter_list:
                        #print(f'Token text: {token.text} POS={token.pos_}') # TESTEN
                        if token.pos_ in pos_tags_to_keep:
                            if token.pos_ == 'NUM':
                                if len(token.text.lower()) == 4:
                                    tokenized_article.append(token.lemma_.lower())
                            else:
                                tokenized_article.append(token.lemma_.lower()) # setting to lowercase
        
        lemmatized_text.append(' '.join(tokenized_article))
        print(f'Preprocessing Art. ID: {counter}')
        counter += 1
    
    return lemmatized_text
   

def read_csv(overview_file):
    df = pd.read_csv(overview_file, encoding='utf-8')
    # df.set_index('id', drop=False)

    # # Check if all columns exist:
    # if 'sub_org' not in df.columns:
    #     df['sub_org'] = np.nan

    # if 'title' not in df.columns:
    #     df['title'] = np.nan

    # if 'date' not in df.columns:
    #     df['date'] = np.nan

    # if 'teaser' not in df.columns:
    #     df['teaser'] = np.nan

    # if 'article_text' not in df.columns:
    #     df['article_text'] = np.nan

    # if 'date_standardized' not in df.columns:
    #     df['date_standardized'] = np.nan

    # if 'text_preprocessed' not in df.columns:
    #     df['text_preprocessed'] = np.nan

    # if 'language_id' not in df.columns:
    #     df['language_id'] = np.nan

    # # set type of columns
    # df["sub_org"]=df["sub_org"].astype(str)
    # df["title"]=df["title"].astype(str)
    # df["date"]=df["date"].astype(str)
    # df["teaser"]=df["teaser"].astype(str)
    # df["article_text"]=df["article_text"].astype(str)
    # df["date_standardized"]=df["date_standardized"].astype(str)
    # df["text_preprocessed"]=df["text_preprocessed"].astype(str)
    # df["language_id"]=df["language_id"].astype(str)

    return df


def get_text_as_list(df, columnname):
    df[columnname] = df[columnname].fillna('') # replace nan values
    text_list = df[columnname].tolist()
       
    return text_list

def add_to_csv(df, col_name, preprocessed_list, overview_file):
    # integrate back into csv
    df[col_name] = preprocessed_list 
    df.to_csv(overview_file, index=False, encoding='utf-8')    
    return True


# auto. language detection
def detect_language(text_list, excep_id):
    lang_ids = []
    counter = 1
    exceptions = 0
    for art in text_list:
        try:
            id = detect(art.strip())
        except:
            logging.info(f'Language undetectable at #: {counter}. Assume default language: {excep_id}.')
            print(f'Language undetectable at #: {counter}. Assume default language.')
            exceptions += 1
            id = excep_id
        print(f'Detecting language of ID: {counter}')
        lang_ids.append(id)
        counter += 1

    print(f'# changed to default language: {exceptions}')
    
    # Stats
    print('Calculating language ID statistics ...')
    stats = Counter(lang_ids)
    print(f'Language Identification stats: {stats}')
    logging.info(f'Language Identification stats: {stats}')

    return lang_ids


def translate():
    ...


def standardize_dates(overview_file, df):
    df.date = df.date.fillna('') # replace nan values
    dates_list = df['date'].tolist()
    standardized_dates = []
    
    for i in range(len(dates_list)):
        if dates_list[i] == 'nan':
            standardized_dates.append('nan')
        else:
            try:
                parsed = str(dateparser.parse(dates_list[i], languages=['de'])) # assume german format
                
                if parsed == 'None':
                    parsed = str(dateparser.parse(dates_list[i], languages=['en'])) # for cases like 2012-06-23
                
                if parsed != 'None':
                    standardized_dates.append(parsed)

                elif len(str(dates_list[i])) > 25 and len(str(dates_list[i])) < 28:
                    #logging.info(f'ID {i+1}: Len -> do not standardize date: {dates_list[i]}')                
                    standardized_dates.append(dates_list[i])
                elif dates_list[i] == '':
                    standardized_dates.append('n.a.')
                else:
                    print(f'Error at ID {i+1} Date: {dates_list[i]}')
                    logging.critical(f'Error at ID {i+1} with date: {dates_list[i]}')
                    standardized_dates.append(dates_list[i])


            except TypeError:            
                print(f'Type error at ID {i+1} Date: {dates_list[i]}')
                logging.critical(f'Type error at pos {i+1} Date: {dates_list[i]}')
                standardized_dates.append(dates_list[i])
    
    # integrate back into csv
    df['date_standardized'] = standardized_dates 
    df.to_csv(overview_file, index=False, encoding='utf-8')  
    print(f'Standardized dates stored in file {overview_file}')  
    
    return df


# Main
def standard_pipeline(overview_file):
    df = read_csv(overview_file)
    
    # Standardize Dates
    # df = standardize_dates(overview_file, df)

    # language identification
    # text_list = get_text_as_list(df, 'article_text')
    text_list = get_text_as_list(df, 'context')

    # lang_id_list = detect_language(text_list, 'de') # ISO 639-1 codes for language ID; assign tag 'de' in case language is not detectable
    # add_to_csv(df, 'language_id', lang_id_list, overview_file)

    # Lemmatization, lowercase, remove stopwords, keep only some POS Tags
    lang_ids = get_text_as_list(df, 'language')

    pos_tags_to_keep = ['NOUN', 'PROPN', 'VERB', 'INTERJ', 'PRON', 'ADJ', 'NUM'] # UPOS tagnames: https://universaldependencies.org/docs/u/pos/
    filter_list = ["ftp", "http", "mailto", "www", "schweiz"]
    
    lemmatized_list = preprocess(text_list, lang_ids, pos_tags_to_keep, filter_list)
    
    add_to_csv(df, 'context_preprocessed', lemmatized_list, overview_file)

standard_pipeline('mediacorpus.csv')
#standard_pipeline('corriere.csv')
