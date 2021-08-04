from string import punctuation
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import os
import re
import numpy as np

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

#train_path = os.path.join("aclImdb", "train")
#test_path = os.path.join(os.getcwd(), "imdb_te.csv")


def load_stop_words():
    stop_words = []
    with open("stopwords.en.txt", "r") as file:
        line = file.readline()
        while line:
            stop_words.append(line.strip())
            line = file.readline()
    return stop_words


def remove_stop_words(text):
    stop_words = load_stop_words()
    translator = str.maketrans('', '', punctuation)
    query_words = text.translate(translator).split()
    result = [word for word in query_words if word not in stop_words]
    return ' '.join(result)


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
    and combine text files under train_path directory into
    imdb_tr.csv. Each text file in train_path should be stored
    as a row in imdb_tr.csv. And imdb_tr.csv should have two
    columns, "text" and label'''
    print("imdb_data_preprocess")
    train_pos = []
    train_neg = []
    train_pos_path = os.path.join(inpath, "pos")
    train_neg_path = os.path.join(inpath, "neg")

    for root, dirs, files in os.walk(train_pos_path):
        for file in files:
            print(os.path.join(root, file))
            with open(os.path.join(root, file), "r", encoding="utf8") as read_file:
                try:
                    text = read_file.readline().lower()
                except:
                    pass
                #text_stripped = re.sub('[()]', " ", text)
                #text_stripped = re.sub('<br /><br />', " ", text_stripped)
                #text_stripped = re.sub(" \d+", " ", text_stripped)
                #text_stripped = remove_stop_words(text_stripped)
                train_pos.append(text)

    dict_pos = {"text": train_pos, "class": np.ones(len(train_pos), dtype=int)}
    df_pos = pd.DataFrame(dict_pos)

    for root, dirs, files in os.walk(train_neg_path):
        for file in files:
            print(os.path.join(root, file))
            with open(os.path.join(root, file), "r", encoding="utf8") as read_file:
                try:
                    text = read_file.readline().lower()
                except:
                    pass
                #text_stripped = re.sub('[()]', "", text)
                #text_stripped = re.sub('<br /><br />', " ", text_stripped)
                #text_stripped = re.sub(" \d+", " ", text_stripped)
                #text_stripped = remove_stop_words(text_stripped)
                train_neg.append(text)

    dict_neg = {"text": train_neg, "class": np.zeros(len(train_neg), dtype=int)}
    df_neg = pd.DataFrame(dict_neg)
    df_tot = pd.concat([df_pos, df_neg], ignore_index=True)
    df_tot.to_csv(os.path.join(outpath, name), index_label="index", index=True)
    return df_tot


if __name__ == "__main__":
    stop_words = load_stop_words()
    imdb_data_preprocess(train_path)
    df_train = pd.read_csv(os.getcwd() + "/imdb_tr.csv")
    X_train = df_train["text"].values.astype('unicode')
    Y_train = df_train["class"].values
    #print(X_train[:3])

    df_test = pd.read_csv(test_path, encoding="ISO-8859-1", skiprows=0)
    X_test = df_test["text"].values.astype('unicode')
    #print("TEST \n%s"%X_test[:3])


    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    unigram_counts = CountVectorizer(min_df=1, stop_words=stop_words, ngram_range=(1, 1), encoding='unicode')
    unigram = unigram_counts.fit_transform(X_train)
    unigram_classifier = SGDClassifier(loss="hinge", penalty="l1")
    unigram_classifier.fit(unigram, Y_train)
    train_predictions = unigram_classifier.predict(unigram)
    accuracy = accuracy_score(train_predictions, Y_train)
    print(accuracy)
    unigram_test = unigram_counts.transform(X_test)
    test_predictions = unigram_classifier.predict(unigram_test)
    with open("unigram.output.txt", "w") as outfile:
        for predict in test_predictions:
            #print(predict)
            outfile.write(str(predict) + "\n")


    '''train a SGD classifier using bigram representation,
        predict sentiments on imdb_te.csv, and write output to
        bigram.output.txt'''
    bigram_counts = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, token_pattern=r'\b\w+\b', min_df=1)
    bigram = bigram_counts.fit_transform(X_train)
    bigram_classifier = SGDClassifier(loss="hinge", penalty="l1")
    bigram_classifier.fit(bigram, Y_train)
    train_predictions = bigram_classifier.predict(bigram)
    accuracy = accuracy_score(train_predictions, Y_train)
    print(accuracy)
    bigram_test = bigram_counts.transform(X_test)
    test_predictions = bigram_classifier.predict(bigram_test)
    with open("bigram.output.txt", "w") as outfile:
        for predict in test_predictions:
            #print(predict)
            outfile.write(str(predict) + "\n")


    '''train a SGD classifier using unigram representation
         with tf-idf, predict sentiments on imdb_te.csv, and write 
         output to unigramtfidf.output.txt'''
    unigram_tfidf = TfidfVectorizer(min_df=1, stop_words=stop_words)
    tfidf = unigram_tfidf.fit_transform(X_train)
    tfidf_classifier = SGDClassifier(loss="hinge", penalty="l1")
    tfidf_classifier.fit(tfidf, Y_train)
    train_predictions = tfidf_classifier.predict(tfidf)
    accuracy = accuracy_score(train_predictions, Y_train)
    print(accuracy)
    tfidf_test = unigram_tfidf.transform(X_test)
    test_predictions = tfidf_classifier.predict(tfidf_test)
    with open("unigramtfidf.output.txt", "w") as outfile:
        for predict in test_predictions:
            #print(predict)
            outfile.write(str(predict) + "\n")


    '''train a SGD classifier using bigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to bigramtfidf.output.txt'''
    bigram_tfidf = TfidfVectorizer(min_df=1, stop_words=stop_words, ngram_range=(1,2))
    tfidf = bigram_tfidf.fit_transform(X_train)
    tfidf_classifier = SGDClassifier(loss="hinge", penalty="l1")
    tfidf_classifier.fit(tfidf, Y_train)
    train_predictions = tfidf_classifier.predict(tfidf)
    accuracy = accuracy_score(train_predictions, Y_train)
    print(accuracy)
    tfidf_test = bigram_tfidf.transform(X_test)
    test_predictions = tfidf_classifier.predict(tfidf_test)
    with open("bigramtfidf.output.txt", "w") as outfile:
        for predict in test_predictions:
            #print(predict)
            outfile.write(str(predict) + "\n")

