import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import svm
from bs4 import BeautifulSoup
import jieba
import os
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

def split_words(row_list):
    #对获取到的文本进行分词处理
    result = []
    for i in row_list:
        if(len(i)<=3):
            result.append(i)
        else:
            seg_list = list(jieba.cut(i))
            result.extend(seg_list)
    return result

def to_str(x):
    #将分词列表转为空格分开的长字符串以进行tdidf计算
    result = ''
    for i in x:
        if(len(i)>=2):
            result+=i
            result+=' '
    return result

def map_flag(x):
    #将n,d,p映射为0,1,2
    flag = 0
    if(x=='n'):
        flag = 0
    elif(x=='d'):
        flag = 1
    else:
        flag = 2
    return flag

def reverse_map_flag(x):
    #将0,1,2映射为n,d,p
    if(x==0):
        flag = 'n'
    elif(x==1):
        flag = 'd'
    else:
        flag = 'p'
    return flag

def find_text(file_name):
    #获取文件中所需要的文本
    try:
        temp_file = open(file_name,'rb')
    except:
        return 'file_not_find'
    try:
        file_data = temp_file.read()
    finally:
        temp_file.close()
    try:
        file_data = file_data.decode('gbk')
    except:
        try:
            file_data = file_data.decode('utf-8')
        except:
            try:
                file_data = file_data.decode("gb18030")
            except:
                try:
                    file_data = file_data.decode("gb2312")
                except:
                    return 'cannot_decode'
    try:
        soup = BeautifulSoup(file_data, "html.parser")
        xx = u'([\u4e00-\u9fa5]+)'
        p = r"(?<=<title>).+?(?=</title>)"
        temp = str(soup.title)
        result = re.findall(p,temp)#得到网页title
        p = r"<a[^>]+?href=.*>([^<]+)<\/a>"
        result.extend(re.findall(p,file_data))#找到超链接后面的文本
        temp = soup.get_text()
        temp = re.findall(xx,temp)#找到整个网页文本中的中文
        result.extend(temp)
        return result
    except:
        return 'error_text'

def get_max(x):
    #得到模型融合结果
    for i,j in enumerate(x):
        if(j)>=0.5:
            return i

trainFile = "./data/file_list_20170430_new.txt"
#训练集文件
pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
data = pd.read_csv(os.path.basename(trainFile),header=None)
os.chdir(pwd)
data.columns = ['index','flag','hash','url']
del data['url']
file_address = './data/file/'

data['text'] = data['hash'].apply(lambda x: find_text(file_address+x))#读取整个文件列表中每个文件的文本
data['word'] = data['text'].apply(split_words)#分词
del data['text']
data['word'] = data['word'].apply(to_str)#分词列表转为以空格分隔的字符串
data['flag'] = data['flag'].apply(map_flag)#将flag变为数值型

trainFile = './data/myRemark.txt'#格式类似file_list_20170430_new.txt
#抽取部分测试集A文件
pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
data2 = pd.read_csv(os.path.basename(trainFile))
os.chdir(pwd)
data2 = data2[['index','result','hash']]
data2.columns = ['index','flag','hash']
file_address = './data/subject1_A/file/'
data2['text'] = data2['hash'].apply(lambda x: find_text(file_address+x))
data2['word'] = data2['text'].apply(split_words)
del data2['text']
data2['word'] = data2['word'].apply(to_str)
data2['flag'] = data2['flag'].apply(map_flag)
data = pd.concat([data,data2])

# max_features=3000,通过在语料库中的词频，挑出来最重要的3000个词来进行判断
tv = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, encoding='gbk',max_features=3000)
tfidf_train_2 = tv.fit_transform(data['word'])
xgbc = XGBClassifier(n_estimators=100,max_depth=5)#xgb分类模型
xgbc.fit(tfidf_train_2,data['flag'])
clf = svm.SVC(kernel='linear', C=1, probability=True)#svm分类模型
clf.fit(tfidf_train_2,data['flag'])
etc = ExtraTreesClassifier(n_estimators=100,max_features=0.8,n_jobs=-1)#extrstree分类模型
etc.fit(tfidf_train_2,data['flag'])
nn = MLPClassifier(solver='lbfgs',alpha=1e-5,random_state=1)#神经网络分类模型
nn.fit(tfidf_train_2,data['flag'])
print('training complete')

trainFile = "./data/subject1_B/file2_list.txt"

pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
online_list = pd.read_csv(os.path.basename(trainFile),header=None)
os.chdir(pwd)

online_list.columns = ['index','hash','url']
file_address = './data/subject1_B/file/'
result1 = pd.DataFrame()
result2 = pd.DataFrame()
result3 = pd.DataFrame()
result4 = pd.DataFrame()
for i in range(25):
    print(i)
    online_list_temp = online_list.iloc[i*10000:(i+1)*10000]
    online_list_temp['text'] = online_list_temp['hash'].apply(lambda x: find_text(file_address+x))
    online_list_temp['word'] = online_list_temp['text'].apply(split_words)
    online_list_temp['word'] = online_list_temp['word'].apply(to_str)
    del online_list_temp['text']
    tfidf_test = tv.transform(online_list_temp['word'])
    temp_result1 = pd.DataFrame(clf.predict_proba(tfidf_test))
    temp_result2 = pd.DataFrame(xgbc.predict_proba(tfidf_test))
    temp_result3 = pd.DataFrame(etc.predict_proba(tfidf_test))
    temp_result4 = pd.DataFrame(nn.predict_proba(tfidf_test))
    result1 = pd.concat([result1,temp_result1])
    result2 = pd.concat([result2,temp_result2])
    result3 = pd.concat([result3,temp_result3])
    result4 = pd.concat([result4,temp_result4])
result1.columns = ['svm_class1','svm_class2','svm_class3']
result1 = result1.reset_index()
del result1['index']
result2.columns = ['xgb_class1','xgb_class2','xgb_class3']
result2 = result2.reset_index()
del result2['index']
result3.columns = ['etc_class1','etc_class2','etc_class3']
result3 = result3.reset_index()
del result3['index']
result4.columns = ['nn_class1','nn_class2','nn_class3']
result4 = result4.reset_index()
del result4['index']
temp = online_list
temp = pd.merge(temp,result1,how='left',left_index=True,right_index=True)
temp = pd.merge(temp,result2,how='left',left_index=True,right_index=True)
temp = pd.merge(temp,result3,how='left',left_index=True,right_index=True)
temp = pd.merge(temp,result4,how='left',left_index=True,right_index=True)
temp = temp[['index','svm_class1','svm_class2','svm_class3','xgb_class1','xgb_class2','xgb_class3','etc_class1','etc_class2','etc_class3','nn_class1','nn_class2','nn_class3']]
temp['merge_class1'] = temp[['svm_class1','xgb_class1','etc_class1','nn_class1']].mean(axis=1)
temp['merge_class2'] = temp[['svm_class2','xgb_class2','etc_class2','nn_class2']].mean(axis=1)
temp['merge_class3'] = temp[['svm_class3','xgb_class3','etc_class3','nn_class3']].mean(axis=1)

temp['merge_result'] = temp[['merge_class1','merge_class2','merge_class3']].apply(get_max,axis=1)

temp['merge_result'] = temp['merge_result'].apply(reverse_map_flag)

temp1 = temp[temp['merge_result']!='n'][['index','merge_result']]

temp1.columns = ['ID','FLAG']

temp1.to_csv('./data/submit.csv',index=False)
