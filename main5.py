import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns;

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
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from k_means_constrained import KMeansConstrained

stemmer = PorterStemmer()

t1 = 0
t2 = 0


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


def date(date):
    p = date.split('-')
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

df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\CAPSTONE\Capstone - Sheet1 (3).csv")


# df = df.drop(df.index[18])

# x = date.split('-')
# y = x[0]+x[1]+x[2]
def get_path2(m, n, date, day):
    a = m[0]
    b = m[1]
    list4 = []
    g = {}

    l = len(n)
    for i in range(l):
        f = dis(n[i][1], n[i][2], a, b)
        g.update(dict({n[i][0]: f}))
    a1 = g.copy()
    sorted_d = dict(sorted(a1.items(), key=operator.itemgetter(1)))
    list3 = [[k, v] for k, v in sorted_d.items()]
    # print(list3)
    date1 = date
    print("Recommended path for Day {}:".format(day))
    for i in range(len(list3)):
        mn = list3[i][0]
        print(f"{i + 1 : <10}{mn : ^25}{date1 : >20}")
        # print(f"{i + 1 : <10}{path[i] : ^25}{date1 : >20}")



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


def get_hotel(a, b, c, x3, start_date):
    g = {}
    h = {}
    list4 = []
    list6 = []
    list7 = []
    list8 = []
    df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\CAPSTONE\HotelsSouthIndia.csv")
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

    for i in range(10):
        list4.append(list3[i])
    list5 = [j for i in list4 for j in i]
    for i in range(len(list5)):
        if type(list5[i]) == tuple:
            h.update(dict({list5[i][0]: list5[i][1]}))
    # print(h)
    s = dict(sorted(h.items(), key=operator.itemgetter(1), reverse=True))
    # print(s)
    list4 = [[k, v] for k, v in s.items()]
    for i in range(5):
        list7.append(list4[i])
    # print(list7)
    print("These are five hotels recommended choose 1:")
    for i in range(len(list7)):
        print(i + 1, '.', list7[i][0])
        list8.append(list7[i][0])
    vb = int(input("Choose any hotel:"))
    nj = vb - 1
    i1 = []
    for i in range(len(df1)):
        if df1.loc[i][10] == list8[nj]:
            fg = df.loc[i][1]
            gh = df.loc[i][2]
            i1.append(fg)
            i1.append(gh)
    date7 = start_date
    day = 1
    for i in range(len(x3)):
        get_path2(i1, x3[i], date7, day)
        date7 = date(date7)
        day = day + 1


def mean_places1(list2, x3, city, start_date):
    lat = 0
    lon = 0
    for i in range(len(list2)):
        lat = lat + list2[i][0]
        lon = lon + list2[i][1]
    lat_mean = lat / len(list2)
    lon_mean = lon / len(list2)
    get_hotel(lat_mean, lon_mean, city, x3, start_date)


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


def date(date):
    p = date.split('-')
    q = str(int(p[0]) + 1)
    r = q + '-' + p[1] + '-' + p[2]
    return r


# print(y)

# P = df1.reindex(columns = ['Places','Latitude','Longitude'])
# a = pd.date_range(y, periods=days)
# print(a)
# for i in a:
#     print(i)
top_places = []
reviews = []
reviews1 = []
# length = len(df)
length = len(df)


# if len(top_places) < days*4:
#     top_places.append
def ranking(city):
    p = 0
    q = 0
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
    for j in range(length):
        if df.loc[j][1] == city:
            for k in range(len(reviews)):
                if len(reviews1) != 0:
                    if reviews[k] == reviews1[0]:
                        top_places.append(df.loc[k + p][0])
                        reviews1.pop(0)
                        break
    # print(top_places)
    return top_places


# print(df2)
def method2():
    start_date = input("Give us date you want to start travelling in the format of dd-mm-yyyy : ")
    city = input("Tell us the city you want to travel : ")
    days = int(input("No. of days you want to travel in the city : "))
    df1 = df[df.City == city]
    # commenprint(df1)

    top_places = ranking(city)

    top_places_final = []
    if len(top_places) > 4 * days:
        for i in range(4 * days):
            top_places_final.append(top_places[i])
    else:
        for i in range(len(top_places)):
            top_places_final.append(top_places[i])

    # print(top_places_final)
    df2 = df1[df1.Places.isin(top_places_final)]
    df2 = df2.reset_index(drop=True)
    P = df2.reindex(columns=['Places', 'Latitude', 'Longitude'])
    # print(P)
    # print(P)
    # if len(top_places) < 4*days:
    #     top_places.extend(city_places)
    # a =df.loc[1]
    # print(a)
    # if a[1] == b:
    # c = a[5]
    # print(c)
    # print(le
    # n(df))
    # print(P.loc[0])
    if (days == 1 or len(top_places) < 4 * days):
        a = 0
        b = 4

    else:
        a = 3
        b = 5
    kmeans = KMeansConstrained(n_clusters=days, size_min=a, size_max=b, init='k-means++', n_init=10, max_iter=300,
                               tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)

    # kmeans = KMeans(n_clusters = days, init ='k-means++')
    kmeans.fit(P[P.columns[1:3]])  # Compute k-means clustering.
    P['cluster_label'] = kmeans.fit_predict(P[P.columns[1:3]])
    # count = P[P['cluster_label'] == 0]['Places'].count()

    centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
    # print(centers)
    labels = kmeans.predict(P[P.columns[1:3]])  # Labels of each point
    P.plot.scatter(x='Latitude', y='Longitude', c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    count_cluster = []

    x3 = []
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
    mean_places1(x4, x3, city, start_date)


city_list = []


def method1():
    with open(r"C:\Users\karth\OneDrive\Desktop\CAPSTONE\Capstone - Sheet1 (3).csv") as readobj:
        a = list(csv.reader(readobj))
        for i in a:
            if i[1] not in city_list:
                city_list.append(i[1])
    start_date = input("Give us date you want to start travelling in the format of dd-mm-yyyy : ")

    city = input("Tell us the city you want to travel : ")
    if (city not in city_list) or (city is None):
        print("Give a valid city")
        method1()
        pass
    else:

        print("Choose places from the below list : \n")
        k = 1
        places_list = []
        for i in a:
            if i[1] == city:
                p = [i[0], i[2], i[3]]
                places_list.append(p)
                print(k, i[0])
                k = k + 1
        places1 = []
        places_count = int(input("\nNo of places you want to go? "))

        print("\nChoose the places by giving the number of place from given list\n")
        places = []
        for i in range(places_count):
            m = int(input())
            places.append(places_list[m - 1])
        # print("\nThese are the chosen places :")
        b = []
        for i in range(len(places)):
            b.append(places[i][0])
        if (places_count % 4 == 0):
            days = int(places_count / 4)
        else:
            v1 = places_count % 4
            v2 = ranking(city)
            if len(v2) > v1:
                for i in range(len(v2)):
                    if (len(b) % 4 == 0):
                        days = int(len(b) / 4)
                        break

                    elif ((v2[i] not in b)):
                        b.append(v2[i])

            # print(b)

        df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\CAPSTONE\Capstone - Sheet1 (3).csv")
        df1 = df[df.City == city]
        df2 = df1[df1.Places.isin(b)]
        df2 = df2.reset_index(drop=True)

        df3 = df2.reindex(columns=['Places', 'Latitude', 'Longitude'])
        places1 = df3.values.tolist()
        df2 = df1[df1.Places.isin(b)]
        df2 = df2.reset_index(drop=True)
        P = df2.reindex(columns=['Places', 'Latitude', 'Longitude'])
        # if len(top_places) < 4*days:
        #     top_places.extend(city_places)
        # a =df.loc[1]
        # print(a)
        # if a[1] == b:
        # c = a[5]
        # print(c)
        # print(le
        # n(df))
        # print(P.loc[0])

        if (days == 1 or len(b) < 4 * days):
            a = 0
            b = 4

        else:
            a = 3
            b = 5
        kmeans = KMeansConstrained(n_clusters=days, size_min=a, size_max=b, init='k-means++', n_init=10, max_iter=300,
                                   tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1)

        # kmeans = KMeans(n_clusters = days, init ='k-means++')
        kmeans.fit(P[P.columns[1:3]])  # Compute k-means clustering.
        P['cluster_label'] = kmeans.fit_predict(P[P.columns[1:3]])
        # count = P[P['cluster_label'] == 0]['Places'].count()
        # count1 = P[P['cluster_label'] == 1]['Places'].count()
        # count2 = P[P['cluster_label'] == 2]['Places'].count()
        centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
        # print(centers)
        labels = kmeans.predict(P[P.columns[1:3]])  # Labels of each point
        # print(labels)
        # P.head(10)
        P.plot.scatter(x='Latitude', y='Longitude', c=labels, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()
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
        mean_places1(x4, x3, city, start_date)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(r'C:\Users\karth\PycharmProjects\chatbot\intents.json') as json_data:
    intents = json.load(json_data)

FILE = r"C:\Users\karth\PycharmProjects\chatbot\data.pth"
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
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

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
                    print("\n Choose an option")
                    print("1. User selects places")
                    print("2. Chatbot suggests places")
                    method = int(input())
                    if (method == 1):
                        method1()
                    elif method == 2:
                        method2()
                    else:
                        print("Choose valid method")
                else:

                    print(f"{bot_name}: {random.choice(intent['responses'])}")

    else:
        print(f"{bot_name}: Sorry, I do not understand...")


