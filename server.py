from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
import string, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import re
from googlesearch import search
from urllib.parse import urlparse
from tld import get_tld
from sklearn.preprocessing import LabelEncoder



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS']='Content-Type'   


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


# helpers...

def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def pre_training_model_for_reviews():
    try:
        df = pd.read_csv('./fraud reviews detection/Preprocessed Fake Reviews Detection Dataset.csv')
        df['text_'] = df['text_'].apply(preprocess)
        return df
    except Exception as e:
        print(e)
        print("Please purify dataset first.")
        exit(0)

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0
    
def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:
        return 0

def google_index(url):
    site = search(url, 5)
    return 1 if site else 0

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

def count_www(url):
    url.count('www')
    return url.count('www')

def count_atrate(url):     
    return url.count('@')

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')

def count_per(url):
    return url.count('%')

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def url_length(url):
    return len(str(url))

def hostname_length(url):
    return len(urlparse(url).netloc)

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def phishing_feature_extraction(url):
    features = []
    features.append(having_ip_address(url))
    features.append(abnormal_url(url))
    features.append(count_dot(url))
    features.append(count_www(url))
    features.append(count_atrate(url))
    features.append(no_of_dir(url))
    features.append(no_of_embed(url))
    features.append(shortening_service(url))
    features.append(count_https(url))
    features.append(count_http(url))
    features.append(count_per(url))
    features.append(count_hyphen(url))
    features.append(count_ques(url))
    features.append(count_equal(url))
    features.append(url_length(url))
    features.append(hostname_length(url))
    features.append(suspicious_words(url))
    features.append(digit_count(url))
    features.append(letter_count(url))
    features.append(fd_length(url))

    tld = get_tld(url, fail_silently=True)
    features.append(tld_length(tld))

    return features
 





# Models tarining...

def reviews_detection():
    df = pre_training_model_for_reviews()
    review_train, _, label_train, _ = train_test_split(df['text_'], df['label'], test_size=0.20)

    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', RandomForestClassifier())
    ])
    pipeline.fit(review_train, label_train)
    return pipeline

def payment_detection():
    data_set = pd.read_csv("./Fraud Payment Detection/dataset.csv")
    data_set.isnull().sum()
    data_set = data_set.dropna()
    data_set.replace(to_replace=['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN'],value=[2,4,1,5,3],inplace=True)

    X = data_set[['type','amount','oldbalanceOrg','newbalanceOrig']]
    Y = data_set.iloc[:,-2]

    model = DecisionTreeClassifier()
    X_tarin, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.2, random_state=42)
    model.fit(X_tarin, Y_train)
    return model

def phishing_website_detection_model():
    dataset = pd.read_csv("./phishing website detection/dataset.csv")


    dataset['use_of_ip'] = dataset['url'].apply(lambda i: having_ip_address(i))
    dataset['abnormal_url'] = dataset['url'].apply(lambda i: abnormal_url(i))
    dataset['google_index'] = dataset['url'].apply(lambda i: google_index(i))
    dataset['count.'] = dataset['url'].apply(lambda i: count_dot(i))
    dataset['count-www'] = dataset['url'].apply(lambda i: count_www(i))
    dataset['count@'] = dataset['url'].apply(lambda i: count_atrate(i))
    dataset['count_dir'] = dataset['url'].apply(lambda i: no_of_dir(i))
    dataset['count_embed_domian'] = dataset['url'].apply(lambda i: no_of_embed(i))
    dataset['short_url'] = dataset['url'].apply(lambda i: shortening_service(i))
    dataset['count-https'] = dataset['url'].apply(lambda i : count_https(i))
    dataset['count-http'] = dataset['url'].apply(lambda i : count_http(i))
    dataset['count%'] = dataset['url'].apply(lambda i : count_per(i))
    dataset['count?'] = dataset['url'].apply(lambda i: count_ques(i))
    dataset['count-'] = dataset['url'].apply(lambda i: count_hyphen(i))
    dataset['count='] = dataset['url'].apply(lambda i: count_equal(i))
    dataset['url_length'] = dataset['url'].apply(lambda i: url_length(i))
    dataset['hostname_length'] = dataset['url'].apply(lambda i: hostname_length(i))
    dataset['sus_url'] = dataset['url'].apply(lambda i: suspicious_words(i))
    dataset['count-digits']= dataset['url'].apply(lambda i: digit_count(i))
    dataset['count-letters']= dataset['url'].apply(lambda i: letter_count(i))
    dataset['fd_length'] = dataset['url'].apply(lambda i: fd_length(i))
    dataset['tld'] = dataset['url'].apply(lambda i: get_tld(i,fail_silently=True))
    dataset['tld_length'] = dataset['tld'].apply(lambda i: tld_length(i))

    lb_make = LabelEncoder()
    dataset["type_code"] = lb_make.fit_transform(dataset["type"])


    X = dataset[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
        'count_dir', 'count_embed_domian', 'short_url', 'count-https',
        'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
        'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
        'count-letters']]
    y = dataset['type_code']


    model = RandomForestClassifier(n_estimators=100,max_features='sqrt')
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)
    model.fit(X_train,y_train)

    return model



# models initializers...

reviews_model = reviews_detection()
payment_model = payment_detection()
phishing_model = phishing_website_detection_model()



# APIs...

@app.route('/predict-review', methods=['POST'])
@cross_origin()
def predict_review():
    # CG means good review and OR means fake review
    text = request.form.get("text")
    if text:
        result = reviews_model.predict([f"{text}"])
        output = "Good Review"
        if result[0]=="OR":
            output="Fake Review"
        return make_response({"isError": False, "result": output})
    return make_response({"isError": True, "result": "Text field is empty"})


@app.route('/predict-payment', methods=['POST'])
@cross_origin()
def predict_payment():
    type = request.form.get("type")
    amount = request.form.get("amount")
    oldAmount = request.form.get("oldAmount")
    newAmount = request.form.get("newAmount")

    if type and amount and oldAmount and newAmount:
        result = payment_model.predict([[int(type), float(amount), float(oldAmount), float(newAmount)]])
        output="Good Transaction"
        if(result[0]==0):
            output="Fake Transaction"
        return make_response({"isError": False, "result": output})
    return make_response({"isError": True, "result": "Fill Empty Fields first"})
       

@app.route('/predict-phishing-site', methods=['POST'])
@cross_origin()
def predict_phising():
    url = request.form.get("url")
    if url:
        features = phishing_feature_extraction(url)
        features = np.array(features).reshape((1,-1))
        result = phishing_model.predict(features)

        output = "This site is Safe. You can use it..."
        if int(result[0])==1.0:
            output="Defacement detected..."
        elif int(result[0])==2.0:
            output="Phishing site detected. Be aware..."
        elif int(result[0])==3.0:
            output="Malware detected in this site..."
        return make_response({"isError": False, "result": output})
    return make_response({"isError": True, "result": "Fill Empty Field first"})




if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000) 
