# Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import json
import pandas as pd

def preprocessing(data_path):
    """
     IN THIS FUNCTION  WE REMOVE PUNCTUATIONS AND DONE LOWER CASE OF THE DOCUMENTS
    :param data_path: 2 columns on this data ,'transcript' and 'url'
    :return: 2 lists, preprocessed transcript  and url 
    """

    # read the data
    ted_talk_data = pd.read_csv(data_path)
    print(ted_talk_data)
    # get the columns
    columns = ted_talk_data.columns

    names = ted_talk_data.url
    all_names_with_title = []
    for i in range(names.__len__()):
        all_names_with_title.append(names[i][26:])


    ted_talk_data['talker_name_and_title'] = all_names_with_title


    # Remove punchuations from the column transcript which contains  our  documents of data .
    ted_talk_data['rmve_punc_data'] = ted_talk_data['transcript'].str.replace('[^\w\s]', '')

    # Change case  of the  documents  to lower case .
    ted_talk_data['rmve_punc_data'] = ted_talk_data['rmve_punc_data'].str.lower()

    # Took  out 2  imp columns  important for us
    transformed_data = ted_talk_data[['rmve_punc_data', 'talker_name_and_title']]

    # Converting  the 2 columns into  lists  for   applying seperately  embedding .
    list_sent = transformed_data['rmve_punc_data'].tolist()
    list_title = transformed_data['talker_name_and_title'].tolist()

    return list_sent,list_title


def model_building(list_sent,num_clusters,model):
    """
     THIS FUNCTION  IS TO APPLY TF-IDF ON OUR DOCUMENT AND APPLIED KMEANS TO OUR TOKENS TO FIND 5 CLUSTERS
    :param list_sent: It contains the document in list format .
    :param num_clusters: defining number of cluster .
    :param model: kmeans model buiding .
    :return: This function return terms : Feature names , cluster :number of clusters , order centroid : all centroids  of clusters .
    """


    # Initialize TFIDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, ngram_range=(1, 3))

    #Apply  our tf-idf  vectorizer  in our  documents

    tfidf_matrix = tfidf_vectorizer.fit_transform(list_sent)

    # terms is just a list of the features used in the tf-idf matrix. This is a vocabulary.

    terms = tfidf_vectorizer.get_feature_names()

    # Apply kmeans  to segregate   all  common  words
    km = KMeans(n_clusters=num_clusters)
    # Fitting our document into  kmeans
    km.fit(tfidf_matrix)
    # Putting  all  cluster into  list
    clusters = km.labels_.tolist()

    # Defining centroids  of every clusters
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    # store our  model
    joblib.dump(km, model)

    return terms,clusters,order_centroids


def model_validation(model,list_title,num_clusters,terms, clusters, order_centroids):
    """
    THIS  FUNCTION WE  ARE LOADING  OUR MODEL  AND  PRINTING 5 CLUSTERS OF SIMILAR SPEAKERS
    :param model: 
    :param list_title: It  contains the list of speaker names with  topics 
    :param num_clusters:
    :param terms: 
    :param clusters:
    :param order_centroids:
    :return:
    """

    # Extract  our model to use
    km = joblib.load(model)
    # Putting our  title data into a dataframe
    last_df = {'title': list_title}
    results = pd.DataFrame(last_df, index=[clusters])

    # This below code will give top 5 clusters of
    # auretors  having similar  speech

    for i in range(num_clusters):
        print("Cluster %d titles:" % i, end='')
        for title in results.ix[i]['title'].values.tolist()[1:num_clusters]:
            print(' %s,' % title, end='')

    # This  will  give  5  clusters ,
    # in each clusters top 5 words which are similar.
    for i in range(num_clusters):
        print("Cluster %d:" % i)
        for ind in order_centroids[i,:num_clusters]:  ##### You can change it  and see  how many values you want to see in each  cluster
            print(' %s' % terms[ind])

if __name__ == '__main__':

    # Preprocessing
    data_path = "E://personal//datasets//transcripts.csv"
    list_sent, list_title = preprocessing(data_path)
    print("Preprocessing completed")


    # Model Building
    num_clusters = 5
    model = 'doc_cluster.pkl'
    terms, clusters, order_centroids = model_building(list_sent,num_clusters,model)
    print("Model Building completed")

    # Model Validation

    model_validation(model, list_title, num_clusters, terms, clusters, order_centroids)
    print("Model Validation completed")