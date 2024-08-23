
#import necessary libraries
import numpy as np
import pandas as pd
import ast
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
import string
stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from flask import *
import mysql.connector
db=mysql.connector.connect(user="root",password="",port='3306',database='Recomanded')
cur=db.cursor()


#define flask
app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

#web-address of index page
@app.route('/')
def index():
    return render_template('index.html')

#web-address of about page
@app.route('/about')
def about():
    return render_template('about.html')


#web-address of login page
@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("userhome.html",myname=data[0][1])
    return render_template('login.html')

#web-address of registration page
@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')

#web-address of view page
@app.route('/view')
def view():
    global a
    dataset = pd.read_csv('flipkart_com-ecommerce_sample.csv')
    # pre_df = dataset
    print(dataset)
    dataset = dataset[:500]
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

def filter_keywords(doc):
    doc=doc.lower()
    stop_free = " ".join([i for i in doc.split() if i not in stop_words])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    word_tokens = word_tokenize(punc_free)
    filtered_sentence = [(lem.lemmatize(w, "v")) for w in word_tokens]
    return filtered_sentence



#web-address of recommendation page
@app.route('/recommendation',methods=['POST','GET'])
def recommendation():
    global x_train,y_train, smd,cosine_sim,indices,titles
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)

        pre_df = pd.read_csv('flipkart_com-ecommerce_sample.csv')

        pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('[]'))
        pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('"'))
        pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.split('>>'))

        #delete unwanted columns
        del_list=['crawl_timestamp','product_url','image',"retail_price","discounted_price","is_FK_Advantage_product","product_rating","overall_rating","product_specifications"]
        pre_df=pre_df.drop(del_list,axis=1)

        smd=pre_df.copy()
        # drop duplicate produts
        smd.drop_duplicates(subset ="product_name",keep = "first", inplace = True)
        smd.shape

        smd['product'] = smd['product_name'].apply(filter_keywords)
        smd['description'] = smd['description'].astype("str").apply(filter_keywords)
        smd['brand'] = smd['brand'].astype("str").apply(filter_keywords)

        smd["all_meta"]=smd['product']+smd['brand']+ pre_df['product_category_tree']+smd['description']
        smd["all_meta"] = smd["all_meta"].apply(lambda x: ' '.join(x))
        smd["all_meta"].head()

        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(smd['all_meta'])

        from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        smd = smd.reset_index()
        titles = smd['product_name']
        indices = pd.Series(smd.index, index=smd['product_name'])
        
        #recommending products regarding product name
        rec = get_recommendations(f1)
        print(rec)
        rec.head()
        print('################')
        print(type(rec))
        print('########################')
        rec = pd.DataFrame(rec)
        print('########################') 
        print(type(rec))
        print(rec.head())

        return render_template('recommendation.html',msg=rec, columns=rec.columns.values, rows=rec.values.tolist())    

    return render_template('recommendation.html')

# smd = smd.reset_index()
# titles = smd['product_name']
# indices = pd.Series(smd.index, index=smd['product_name'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]

    
if __name__=='__main__':
    app.run(debug=True)