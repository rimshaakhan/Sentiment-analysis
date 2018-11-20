from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import nltk
import numpy as np
from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
from flask import Flask, render_template, redirect, url_for
from flask import request



lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('positive.review.txt').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('negative.review.txt').read())
negative_reviews = negative_reviews.findAll('review_text')



def tokens_preprocessing(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens



vocabulary = {}
index = 0
positive_tokenized = []
negative_tokenized = []
original = []

for review in positive_reviews:
    original.append(review.text)
    tokens = tokens_preprocessing(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = index
            index += 1

for review in negative_reviews:
    original.append(review.text)
    tokens = tokens_preprocessing(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = index
            index += 1

def token_vector(tokens, label):
    x = np.zeros(len(vocabulary) + 1) # last element is for the label
    for t in tokens:
        i = vocabulary[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(vocabulary) + 1))
i = 0
for tokens in positive_tokenized:
    z = token_vector(tokens, 1)
    data[i,:] = z
    i += 1

for tokens in negative_tokenized:
    z = token_vector(tokens, 0)
    data[i,:] = z
    i += 1

# shuffle the data and create train/test splits

original, data = shuffle(original, data)


X = data[:,:-1]
Y = data[:,-1]


Xtrain = X[:-50,]
Ytrain = Y[:-50,]
Xtest = X[-50:,]
Ytest = Y[-50:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))

def token_vector2(tokens):
    x = np.zeros(len(vocabulary))
    for t in tokens:
        if t in vocabulary:
            i = vocabulary[t]
            x[i] += 1

    x = x / x.sum() 
    return x


def update_one(comment):
    f=open('comments.txt','a')
    f.write('\n'+'<review><review_text>'+comment+'</review_text></review>')
    f.close()

    comments_user = BeautifulSoup(open('comments.txt').read())
    comments_user = comments_user.findAll('review_text')
    str= []
    rating1=[]
    i=0
    for review in comments_user:
        comment_1=(review.text)
        str.append(comment_1)
        sentiment=positive_negative(review.text)
        rating1.append(sentiment)
        i=i+1
        
    return str[i-1], str[i-2], str[i-3], str[i-4], str[i-5], rating1[i-1], rating1[i-2], rating1[i-3], rating1[i-4], rating1[i-5]


def info_one():
    comments_user = BeautifulSoup(open('comments.txt').read())
    comments_user = comments_user.findAll('review_text')
    str= []
    rating1=[]
    i=0
    for review in comments_user:
        abcd=(review.text)
        str.append(abcd)
        a=positive_negative(review.text)
        rating1.append(a)
        i=i+1
        
    return str[i-1], str[i-2], str[i-3], str[i-4], str[i-5], rating1[i-1], rating1[i-2], rating1[i-3], rating1[i-4], rating1[i-5]


def positive_negative(comment):
    tokens_user = tokens_preprocessing(comment)
    z=token_vector2(tokens_user)
    z=z.reshape(1,-1)
    sentiment=model.predict(z)
    if sentiment==1:
        return "POSITIVE"
    
    if sentiment==0:
        return "NEGATIVE"
 
def update_two(comment):
    f=open('commentstwo.txt','a')
    f.write('\n'+'<review><review_text>'+comment+'</review_text></review>')
    f.close()

    comments_usertwo = BeautifulSoup(open('commentstwo.txt').read())
    comments_usertwo = comments_usertwo.findAll('review_text')
    i=0
    rating2=[]
    str=[]
    for review in comments_usertwo:
        abcd=(review.text)
        str.append(abcd)
        sentiment=positive_negative(review.text)
        rating2.append(sentiment)
        i=i+1
        

    return str[i-1], str[i-2], str[i-3], str[i-4], str[i-5], rating2[i-1], rating2[i-2], rating2[i-3], rating2[i-4], rating2[i-5]


def info_two():
    comments_usertwo = BeautifulSoup(open('commentstwo.txt').read())
    comments_usertwo = comments_usertwo.findAll('review_text')

    i=0
    rating2=[]
    str=[]
    for review in comments_usertwo:
        abcd=(review.text)
        str.append(abcd)
        a=positive_negative(review.text)
        rating2.append(a)
        i=i+1
        

    return str[i-1], str[i-2], str[i-3], str[i-4], str[i-5], rating2[i-1], rating2[i-2], rating2[i-3], rating2[i-4], rating2[i-5]



def rating_one():
    comments_user = BeautifulSoup(open('comments.txt').read())
    comments_user = comments_user.findAll('review_text')

    positive=1.0
    negative=1.0
    r=1.0
    for review in comments_user:
        sentiment=positive_negative(review.text)
        
        if sentiment=='POSITIVE':
            positive=positive+1

        if sentiment=='NEGATIVE':
            negative=negative+1
        
    r=float(positive+negative)
    total=float(positive+negative)
    r=float(positive/r)
    
    csat=float(positive/total)
    csat=csat*100
    dsat=float(negative/total)
    dsat=dsat*100
    r=float(r*5)
    r=round(r,2)
    csat=round(csat,2)
    dsat=round(dsat,2)
    return r, csat, dsat

def rating_overall():
    comments_user1=[]
    comments_user2=[]
    
    comments_user1 = BeautifulSoup(open('comments.txt').read())
    comments_user1 = comments_user1.findAll('review_text')
    comments_user2 = BeautifulSoup(open('commentstwo.txt').read())
    comments_user2 = comments_user2.findAll('review_text')


    positive=1.0
    negative=1.0
    r=1.0
    for review in comments_user1:
        sentiment=positive_negative(review.text)
        if sentiment=='POSITIVE':
            positive=positive+1

        if sentiment=='NEGATIVE':
            negative=negative+1

    for review in comments_user2:
    
        sentiment=positive_negative(review.text)
        if sentiment=='POSITIVE':
            positive=positive+1

        if sentiment=='NEGATIVE':
            negative=negative+1
    


    r=float(positive+negative)
    total=float(positive+negative)
    r=float(positive/r)
    r=float(r*5)
    r=round(r,2)
    return r




def rating_two():
    comments_usertwo = BeautifulSoup(open('commentstwo.txt').read())
    comments_usertwo = comments_usertwo.findAll('review_text')

    positive=1.0
    negative=1.0
    r=1.0
    for review in comments_usertwo:
        sentiment=positive_negative(review.text)
        if sentiment=='POSITIVE':
            positive=positive+1

        if sentiment=='NEGATIVE':
            negative=negative+1
        
    r=float(positive+negative)
    total=float(positive+negative)
    r=float(positive/r)
    
    csat=float(positive/total)
    csat=csat*100
    dsat=float(negative/total)
    dsat=dsat*100
    r=float(r*5)
    r=round(r,2)
    csat=round(csat,2)
    dsat=round(dsat,2)
    return r, csat, dsat




app = Flask(__name__)
@app.route('/')
def home():
    rating=rating_overall()
    r1, csat, dsat=rating_one()
    r2, csat, dsat=rating_two()
    return render_template('home.html', rating=rating, r1=r1, r2=r2)

@app.route('/employeeone/')
def employeeone():
    r1, x, y=rating_one()
    rating=rating_overall()
    r2, a, b=rating_two()
    return render_template('employeeone.html', r1=r1, rating=rating, r2=r2)


@app.route('/knowyouremployee/')
def knowyouremployee():
    text1, text2, text3, text4, text5, r1, r2, r3, r4, r5=info_one()
    r, csat, dsat=rating_one()
    return render_template('knowyouremployee.html', text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, csat=csat, dsat=dsat, r=r)


@app.route('/employeeone/', methods = ['POST'])
def employeeone_post():
  
   text=request.form['text']
   text=text.decode('utf-8','ignore').encode("utf-8")

   text1, text2, text3, text4, text5, r1, r2, r3, r4, r5=update_one(text)
   r, csat, dsat=rating_one()
   return render_template('knowyouremployee.html', text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, csat=csat, dsat=dsat, r=r)


@app.route('/employeetwo/')
def employeetwo():
    r1, x, y=rating_one()
    rating=rating_overall()
    r2, b, c=rating_two()
    return render_template('employeetwo.html', r1=r1, rating=rating, r2=r2)

@app.route('/knowyouremployee2/')
def knowyouremployee2():
    text1, text2, text3, text4, text5, r1, r2, r3, r4, r5=info_two()
    r, csat, dsat=rating_two()
    return render_template('knowyouremployee2.html', text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, csat=csat, dsat=dsat, r=r)

@app.route('/employeetwo/', methods = ['POST'])
def employeetwo_post():
  
   text=request.form['text']
   text1, text2, text3, text4, text5, r1, r2, r3, r4, r5=update_two(text)
   r, csat, dsat=rating_two()
   return render_template('knowyouremployee2.html', csat=csat, dsat=dsat, r=r, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, r1=r1, r2=r2, r3=r3, r4=r4, r5=r5)
   

if __name__ == '__main__':
   app.run(debug = True)
