# tfidf-kmeans
This repo is for finding document similarity using tfidf and  kmeans , we used  ted talk dataset  from keggle 

In this small  project  i tried  to create document similarity using tfidf and kmean clustering .

I used  TAD Talk's transcript dataset . In the dataset there are 2 columns , 
"URL" : It is having speakers name embedded with  topic of his talk .
"transcript" It is having the whole corpus of individual talk  in text format . 

I added  a script  , tf_idf_kmeans.py file  , which contains 3 functions ,

1st, preprocessing , in this i converted  the  transcript column  into lowercase  and remove punctuations from them ,from the URL column 
i extracted  only  speakername_topic  and appended into  a new column , then converted , both preprocessed columns into 2 lists for processing .

2nd , model_building , in this i applied tfidf with unigram into  transcript list to find important words from the corpus of documents .
then build kmeans clusterring model   on the tokens of important words  to  cluster into 5  similar , and saved the model . 

3rd, model_validation ,  in this  i printed 5 clusters  of common 5 speakers  , along with that  , common 5 important words in every clusters.

Area of improvement  ,  we can  use  bigram /trigram  on  tfidf   ,  we can use  word2vec model  for better  results .

I dont  known much about those  talks , so cannot evaluate  much  , but  below are  the results  i  obtained :


##############################  MOST  SIMILAR  SPEAKERS AND THERE TOPICS BY CLUSTERS


Cluster 0 
titles: eve_ensler_on_happiness_in_body_and_soul
, stew_says_black_men_ski
, jacqueline_novogratz_on_patient_capitalism
, isabel_allende_tells_tales_of_passion

,Cluster 1 
titles: david_deutsch_on_our_place_in_the_cosmos
, richard_dawkins_on_our_queer_universe
, robert_neuwirth_on_our_shadow_cities
, martin_rees_asks_is_this_our_final_century

,Cluster 2
titles: al_gore_on_averting_climate_crisis
, tony_robbins_asks_why_we_do_what_we_do
, julia_sweeney_on_letting_go_of_god
, rick_warren_on_a_life_of_purpose

,Cluster 3
titles: hans_rosling_shows_the_best_stats_you_ve_ever_seen
, cameron_sinclair_on_open_source_architecture
, larry_brilliant_wants_to_stop_pandemics
, amy_smith_shares_simple_lifesaving_design

,Cluster 4
titles: dan_dennett_s_response_to_rick_warren
, jeff_han_demos_his_breakthrough_touchscreen
, nicholas_negroponte_on_one_laptop_per_child
, sirena_huang_dazzles_on_violin

#################################################### MOST COMMON WORDS BY  CLUSTERS



,Cluster 0:
 women
 men
 woman
 said
 world
 
Cluster 1:
 water
 new
 space
 city
 world
 
Cluster 2:
 said
 life
 got
 did
 love
 
Cluster 3:
 world
 percent
 countries
 need
 country
 
Cluster 4:
 actually
 things
 right
 youre
 little
