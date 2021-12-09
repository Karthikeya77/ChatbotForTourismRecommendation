import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns;
from fuzzywuzzy import fuzz

sns.set()
import csv
from geopy import distance
from math import sin, cos, sqrt, atan2, radians
import operator
from k_means_constrained import KMeansConstrained
import torch

import torch.nn as nn
import json
import random

import nltk
import datetime
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from k_means_constrained import KMeansConstrained 
from flask import Flask, render_template, request ,redirect,url_for
app = Flask(__name__,template_folder=r"C:\Users\karth\OneDrive\Desktop\Capstone")
app.static_folder = r'C:\Users\karth\OneDrive\Desktop\Capstone\static'   # Flask constructor
stemmer = PorterStemmer()

t1 = 0
t2 = 0
start_date=""







def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


def date1(d):
    p = d.split('-')
    q = str(int(p[0]) + 1)
    r = q + '-' + p[1] + '-' + p[2]
    return r



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


# from clustering.equal_groups import EqualGroupsKMeans

df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\Capstone\codes\Capstone.csv")


# df = df.drop(df.index[18])

# x = date.split('-')
# y = x[0]+x[1]+x[2]


def get_path2(m, n, date, day):
    a = m[0]
    b = m[1]
    
    list6=[]
    
    #print(n)
    list4 = []
    g = {}
    g2={}
    y=[]
    l = len(n)
    for i in range(l):
        f = dis(n[i][1], n[i][2], a, b)
        g.update(dict({f: n[i]}))
    a1 = g.copy()
    sorted_d = dict(sorted(a1.items(), key=operator.itemgetter(0)))
    #print(sorted_d)
    list3 = [[k, v] for k, v in sorted_d.items()]
    f1=list3[0][1]
    del list3[0]
    a3=f1[1]
    b3=f1[2]
    for i in range(len(list3)):
        g1= dis(list3[i][1][1], list3[i][1][2], a3, b3)
        g2.update(dict({g1: list3[i]}))
    a4 = g2.copy()
   
    sorted_d1 = dict(sorted(a4.items(), key=operator.itemgetter(0)))
    list6 = [v for k, v in sorted_d1.items()]
    list6.insert(0,f1)
    date1 = date
    #print("Recommended path for Day {}:".format(day))
    for i in range(len(list6)):
        mn = list6[i][0]
        if(i>0):
            mn=list6[i][1][0]

        y.append((f"{i + 1 : <10}{mn : ^25}"))
        y.append("@")
    res1=""
    res1=res1.join(y)
    
    
    return res1

def dis(a, b, c, d):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(a)
    lon1 = radians(b)
    lat2 = radians(c)
    lon2 = radians(d)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c
    return d
@app.route("/get/h1")
def main_hotel1():

    x=request.args.get('msg')
   
    x=x.strip()
    vj=x.replace("HT","")
    
    df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\Capstone\codes\HotelsSouthIndia.csv")
    global df1

    df1 = df[df.city == city]
    df1 = df1.reset_index(drop=True)
    
    x4=x3.copy()

    i1 = []
    gh=""
    for i in range(len(df1)):
        if df1.loc[i][10] == vj:
           
            fg = df.loc[i][1]
            gh = df.loc[i][2]
            i1.append(fg)
            i1.append(gh)
    
    date7 = start_date
    day = 1
    res=""
    print(x3)
    
    for i in range(len(x3)):
        op=get_path2(i1, x3[i], date7, day)
       
        res=res+op+"$"
        
        date7 = date1(date7)
        day = day + 1
    res=res+"PT"

    return res

def get_hotel(a, b, c, x3, start_date):
    g = {}
    h = {}
    list4 = []
    list6 = []
    list7 = []
    list8 = []
    df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\Capstone\codes\HotelsSouthIndia.csv")
    global df1
    df1 = df[df.city == c]
    df1 = df1.reset_index(drop=True)
    l = len(df1)
    for i in range(l):
        f = dis(df1.loc[i][1], df1.loc[i][2], a, b)
        g.update(dict({(df1.loc[i][10], df1.loc[i][9]): f}))
    # print(g)
    a1 = g.copy()
    sorted_d = dict(sorted(a1.items(), key=operator.itemgetter(1)))
    # print(sorted_d)
    list3 = [[k, v] for k, v in sorted_d.items()]
    # print(list3)
    if len(list3) >= 10:
        for i in range(10):
            list4.append(list3[i])
    else:
        list4 = list3
    list5 = [j for i in list4 for j in i]
    for i in range(len(list5)):
        if type(list5[i]) == tuple:
            h.update(dict({list5[i][0]: list5[i][1]}))
    # print(h)
    s = dict(sorted(h.items(), key=operator.itemgetter(1), reverse=True))
    # print(s)
    list4 = [[k, v] for k, v in s.items()]
    if len(list4) >= 5:
        for i in range(5):
            list7.append(list4[i])
    else:
        list7 = list4
    # print(list7)
    

    res12=""
    #print("These are five hotels recommended choose 1:")
    for i in range(len(list7)):
        #print(i + 1, '.', list7[i][0])
        list8.append(list7[i][0]+"HT")
        list8.append(" ")
    #main_hotel1(userText,x3,start_date)
    
    return res12.join(list8)

def mean_places1(userText,list2, x3, city, start_date):
    lat = 0
    lon = 0
    for i in range(len(list2)):
        lat = lat + list2[i][0]
        lon = lon + list2[i][1]
    lat_mean = lat / len(list2)
    lon_mean = lon / len(list2)
    ji=get_hotel(lat_mean, lon_mean, city, x3, start_date)
    return ji


def mean_places(list1, city):
    lat = 0
    lon = 0
    list2 = []
    for i in range(len(list1)):
        lat = lat + (list1[i][1])
        lon = lat + (list1[i][2])
    lat_mean = lat / len(list1)
    lon_mean = lon / len(list1)
    list2.append([lat_mean, lon_mean])
    return list2

length = len(df)


# if len(top_places) < days*4:
#     top_places.append
def ranking(city):
    p = 0
    q = 0
    places = []
    reviews = []
    for i in range(length):
        if df.loc[i][1] == city:
            no_of_reviews = str(df.loc[i][5])
            if ',' in no_of_reviews:
                no_of_reviews = no_of_reviews.replace(',', '')
            no_of_reviews = int(no_of_reviews)
            reviews.append(no_of_reviews)
            if q == 0:
                p = i
                q = 1
    reviews1 = reviews.copy()
    reviews1.sort(reverse=True)
    top_places=[]
    for j in range(length):
        if df.loc[j][1] == city:
            
            for k in range(len(reviews)):
                if len(reviews1) != 0:
                    if reviews[k] == reviews1[0]:
                        top_places.append(df.loc[k + p][0])
                        reviews1.pop(0)
                        break
    
    return top_places



# print(df2)
@app.route("/get/h")

def method2():
    userText = request.args.get('msg')
    
    Lk=userText.split(" ")
    #print(Lk)
    userText=""
    
    global start_date
    start_date=Lk[0]
    l1=start_date.split('-')
    t2=l1[2]
    l1[2]=l1[0]
    l1[0]=t2
    start_date=l1[0]+'-'+l1[1]+'-'+l1[2]
    global city
    city=Lk[1]
   
    flag=0
    
   
    flag1=1
    for i in Lk[2]:
        if(i.isalpha()):
            flag1=0
            break;
        elif(i.isnumeric()):
            flag1=1
        else:
            flag1=0;
            break;
        
    if(flag1==1 and Lk[2]):
        days=int(Lk[2])
    else:
        return "Invalid No of Days"+"DayINV"
        
   
    with open(r"C:\Users\karth\OneDrive\Desktop\Capstone\codes\Capstone.csv") as readobj:
        a = list(csv.reader(readobj))
        for i in a:
            if i[1] not in city_list:
                city_list.append(i[1])
  
        for i in city_list:
            Ratio = fuzz.ratio(city.lower(),i.lower())
            if(Ratio>90):
                flag=1
            
                city=i
                break;
    if(flag==0):
       return "Invalid City"+"CityErr"
   
    if(1):
       
        df1 = df[df.City == city]
       
        top_places = ranking(city)

        top_places_final = []
        # print(top_places)
        if len(top_places) > 4 * days:
            for i in range(4 * days):
                top_places_final.append(top_places[i])
        else:
            for i in range(len(top_places)):
                top_places_final.append(top_places[i])

        # print(top_places_final)
        df2 = df1[df1.Places.isin(top_places_final)]
        df2 = df2.reset_index(drop=True)
        pp = df2.reindex(columns=['Places', 'Latitude', 'Longitude'])
        
        if days == 1 or len(top_places) < 4 * days:
            a = 1
            b = 4

        else:
            a = 3
            b = 5
        kmeans = KMeansConstrained(n_clusters=days, size_min=a, size_max=b, init='k-means++', n_init=10, max_iter=300,
                                   tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)

        # kmeans = KMeans(n_clusters = days, init ='k-means++')
        try:            
            kmeans.fit(pp[pp.columns[1:3]])  # Compute k-means clustering.
        except ValueError:
            days=len(top_places)//3
            a=2
            b=4
            kmeans = KMeansConstrained(n_clusters=days, size_min=a, size_max=b, init='k-means++', n_init=10, max_iter=300,
                                   tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)
            kmeans.fit(pp[pp.columns[1:3]])
        pp['cluster_label'] = kmeans.fit_predict(pp[pp.columns[1:3]])
        # count = P[P['cluster_label'] == 0]['Places'].count()

        centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
        # print(centers)
        labels = kmeans.predict(pp[pp.columns[1:3]])  # Labels of each point
        pp.plot.scatter(x='Latitude', y='Longitude', c=labels, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

        count_cluster = []

        global x3
        print(2)
        x3=[]
      
        for i in range(days):
            a = pp[pp['cluster_label'] == i]['Places'].values.tolist()

            count_cluster.append(a)
       
    # print(count_cluster)
        for i in range(len(count_cluster)):
            df4 = df1[df1.Places.isin(count_cluster[i])]
            df4 = df4.reset_index(drop=True)
            
        # print(df2)
            G = df4.reindex(columns=['Places', 'Latitude', 'Longitude'])
            x1 = G.values.tolist()
            x3.append(list(x1))
       
        for i in range(len(x3)):
            x4 = mean_places(x3[i], city)
   
        dfh=mean_places1(userText,x4, x3, city, start_date)
    #print(dfh)
    
    
        return dfh


city_list = []

@app.route("/get/p2")
def method1():
    with open(r"C:\Users\karth\OneDrive\Desktop\Capstone\codes\Capstone.csv") as readobj:
        a = list(csv.reader(readobj))
        for i in a:
            if i[1] not in city_list:
                city_list.append(i[1])
    userText = request.args.get('msg')
    
    Lk=userText.split(" ")
    userText=""
    
    global start_date
    start_date=Lk[0]
    global city
    
    city=Lk[1]
  
    flag=0
    for i in city_list:
        Ratio = fuzz.ratio(city.lower(),i.lower())
        if(Ratio>85):
            flag=1
            
            city=i
            break;
    if(flag==0):
       return "Invalid City"+"CityErr"
   
    if(1):
        
        k = 0
        places_list = []
        f=""
        for i in a:
            if i[1] == city:
                p = [i[0], i[2], i[3]]
                places_list.append(p)
               
                f=f+i[0]+"@"
        return f+"p2"
@app.route("/mainmethod1")
def mainmethod1():    
        userText = request.args.get('msg')
    
        Lk=userText.split("@")   
        
        places=[]
        for i in range(len(Lk)-1):
            places.append(Lk[i])
        places_count=len(places)
        # print("\nThese are the chosen places :")
        b = []
        for i in range(len(places)):
            b.append(places[i])

        df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\Capstone\codes\Capstone.csv")
        df1 = df[df.City == city]
        df2 = df1[df1.Places.isin(b)]
        df2 = df2.reset_index(drop=True)

        df3 = df2.reindex(columns=['Places', 'Latitude', 'Longitude'])
        places1 = df3.values.tolist()
        df2 = df1[df1.Places.isin(b)]
        df2 = df2.reset_index(drop=True)
        P = df2.reindex(columns=['Places', 'Latitude', 'Longitude'])

        if places_count % 4 == 0:
            days = int(places_count / 4)
            if places_count == 4:
                kmeans = KMeansConstrained(n_clusters=days, size_min=4, size_max=4, init='k-means++', n_init=10,max_iter=300,
                                           tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)
            else:
                kmeans = KMeansConstrained(n_clusters=days, size_min=3, size_max=5, init='k-means++', n_init=10,max_iter=300,
                                       tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)
            kmeans.fit(P[P.columns[1:3]])  # Compute k-means clustering.

        else:
            days = int(places_count/4)+1
            if places_count == 1:
                kmeans = KMeansConstrained(n_clusters=days, size_min=1, size_max=1, init='k-means++', n_init=10,max_iter=300,
                                           tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)
            elif places_count == 2:
                kmeans = KMeansConstrained(n_clusters=days, size_min=2, size_max=2, init='k-means++', n_init=10,
                                           max_iter=300,
                                           tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)
            elif places_count == 3:
                kmeans = KMeansConstrained(n_clusters=days, size_min=3, size_max=3, init='k-means++', n_init=10,
                                           max_iter=300,
                                           tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)
            else:
                kmeans = KMeansConstrained(n_clusters=days, size_min=1, size_max=4, init='k-means++', n_init=10,max_iter=300,
                                       tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)
            kmeans.fit(P[P.columns[1:3]])  # Compute k-means clustering.

        P['cluster_label'] = kmeans.fit_predict(P[P.columns[1:3]])
        centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
        # print(centers)
        labels = kmeans.predict(P[P.columns[1:3]])  # Labels of each point
        # print(labels)
        # P.head(10)
        global x3
        print(1)
        x3 = []
        count_cluster = []
        for i in range(days):
            a = P[P['cluster_label'] == i]['Places'].values.tolist()

            count_cluster.append(a)
        # print(count_cluster)
        for i in range(len(count_cluster)):
            df4 = df1[df1.Places.isin(count_cluster[i])]
            df4 = df4.reset_index(drop=True)
            # print(df2)
            G = df4.reindex(columns=['Places', 'Latitude', 'Longitude'])
            x1 = G.values.tolist()
            x3.append(list(x1))
            for i in range(len(x3)):
                x4 = mean_places(x3[i], city)
        dfh=mean_places1(userText,x4, x3, city, start_date)
        return dfh


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(r'C:\Users\karth\OneDrive\Desktop\Capstone\codes\intents.json') as json_data:
    intents = json.load(json_data)

FILE = r"C:\Users\karth\OneDrive\Desktop\Capstone\codes\data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Capstone-bot"

@app.route('/cool_form', methods=['GET', 'POST'])
def cool_form():
    

    # show the form, it wasn't submitted
    return render_template('index.html')
@app.route('/Homepage.html')
def Homepage():
    print('success')

    # show the form, it wasn't submitted
    return render_template("Homepage.html")
@app.route('/Login_Register')
def Login_Register():
    print('success')

    # show the form, it wasn't submitted
    return render_template("Login_Register.html")
@app.route("/")
def home1():
    return render_template("Login_Register.html")
@app.route("/index1")
def index1():
    return render_template("index.html")
@app.route("/profile.html")
def profile():
    return render_template("profile.html")    
@app.route("/get")
def chat():
    while True:
        userText = request.args.get('msg')
        sentence =userText 
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if tag == "recommend":
                        return tag
                    else:

                        x=f"{random.choice(intent['responses'])}"
                        x=x+"Res"
                        return x

        else:
             x="Sorry, I do not understand..."+"NoRes"
             return x
if __name__=='__main__':
   app.run()

