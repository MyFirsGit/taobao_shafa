#!-coding= utf-8-!
import re
import time
import requests
import pandas as pd
from retrying import retry
from concurrent.futures import ThreadPoolExecutor
import xlwt
import matplotlib

#开始计时
start = time.clock()

#plist 为1-100页的URL的编号num
plist = []

for i in range(1,101):
    j = 44 * (i-1)
    plist.append(j)

listno = plist
datatmsp = pd.DataFrame(columns=[])#创建表
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'}
#
# while True:
#     #设置最大重试次数
#     @retry(stop_max_attempt_number = 8)
#     def network_programming(num):
#         url = 'https://s.taobao.com/search?q=%E6%B2%99%E5%8F%91&imgfile=&commend=all&' \
#           'ssid=s5-e&search_type=item&sourceId=tb.index&spm=a21bo.2017.201856-taobao' \
#              '-item.1&ie=utf8&initiative_id=tbindexz_20170306&bcoffset=6&ntoffset=6&p4p' \
#              'pushleft=1%2C48&s=' + str(num)
#         # url = 'https://s.taobao.com/search?q=%E6%96%87%E8%83%B8&imgfile=&js=1&stats_click' \
#         #      '=search_radio_all%3A1&initiative_id=staobaoz_20180612&ie=utf8&bcoffset=3&n' \
#         #      'toffset=3&p4ppushleft=1%2C48&s=' + str(num)
#         web = requests.get(url, headers=headers)
#         web.encoding = 'urf-8'
#         return web
#
#
#     #多线程
#     def multithreading():
#         #每次爬取未成功爬取的页
#         number = listno
#         event = []
#         with ThreadPoolExecutor(max_workers=10) as executor:
#             for result in executor.map(network_programming, number,chunksize=10):
#                 event.append(result)
#
#         return event
#
#
#     listpg = []
#     event = multithreading()
#     for i in event:
#         json = re.findall('"auctions":(.*?),"recommendAuctions"', i.text)
#
#         if len(json):
#             table = pd.read_json(json[0])
#             datatmsp = pd.concat([datatmsp,table], axis=0,ignore_index=True)
#
#             pg = re.findall('"pageNum":(.*?),"p4pbottom_up"', i.text)[0]
#             listpg.append(pg)#这里记录每次爬取成功的页码
#
#     lists = []
#     for a in listpg:
#         b = 44 * (int(a) -1)
#         lists.append(b)#将爬取成功的页码转为url中的num
#
#     listn = listno
#     listno = []#将本次爬取失败的页记录到列表中，用于循环爬取
#     for p in listn:
#         if p not in lists:
#             listno.append(p)
#
#     if len(listno) == 0:
#         break
#
# #导出数据
# datatmsp.to_excel('./data/datatmsp.xls', index=False)
# end = time.clock()
#
# print("爬取完成 用时： ",end - start,'s')

# 数据清洗和处理
datatmsp = pd.read_excel('./data/datatmsp.xls')

# 数据缺失值分析

# 数据缺失处理
import missingno as msno
msno.bar(datatmsp.sample(len(datatmsp)),figsize=(10,4))
# 删除缺失值过半的列
half_count = len(datatmsp)/2
datatmsp = datatmsp.dropna(thresh=half_count,axis=1)
datatmsp = datatmsp.drop_duplicates()# 删除重复行

# 对标题/区域/价格/销量进行分析
data = datatmsp[['item_loc','raw_title','view_price','view_sales']]
#data.head()#默认查看前5行数据

#对item_loc列的省份和城市进行拆分
data['province'] = data.item_loc.apply(lambda x: x.split()[0])

# 直辖市的省份和城市相同 所以根据字符长度进行判断
data['city'] = data.item_loc.apply(lambda x: x.split()[0] if len(x) < 4 else x.split()[1])
#data.dtypes

data['sales'] = data.view_sales.apply(lambda x:x.split('人')[0])

# 查看各列数据类型
#data.dtypes

data['sales'] = data.sales.astype('int')#转换数据类型

list_col = ['province','city']
for i in list_col:
    data[i] = data[i].astype('category')

# 删除不用的列
data = data.drop(['item_loc','view_sales'],axis=1)

title = data.raw_title.values.tolist()

import jieba
title_s = []
for line in title:
    title_cut = jieba.lcut(line)
    title_s.append(title_cut)
#print(title_s)

# 使用停用表删除不需要的单词
stopwords = pd.read_excel('./data/stopwords.xlsx')
stopwords = stopwords.stopword.values.tolist()

title_clean = []
for line in title_s:
    line_clean = []
    for word in line:
        if word not in stopwords:
            line_clean.append(word)
    title_clean.append(line_clean)


# 去重，统计每个词语的个数
title_clean_dist = []
for line in title_clean:
    line_dist = []
    for words in line:
        if words not in line_dist:
            line_dist.append(words)
    title_clean_dist.append(line_dist)

# 将所有次转换成list
allwords_clean_dist = []
for line in title_clean_dist:
    for word in line:
        allwords_clean_dist.append(word)


# 将所有词语转换为数据表
dr_allwords_clean_dist = pd.DataFrame({'allwords':allwords_clean_dist})
word_count = dr_allwords_clean_dist.allwords.value_counts().reset_index()
word_count.columns = ['word','count']
#print(word_count.head())

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.misc import imread
plt.figure(figsize=(20,10))
pic = imread("./image/shafa.jpg")
w_c = WordCloud(font_path="./data/DroidSansFallbackFull.ttf",background_color= 'white',mask=pic,max_font_size=60,
                margin=1)
wc = w_c.fit_words({x[0]:x[1] for x in word_count.head(100).values})

plt.imshow(wc,interpolation='bilinear')
#plt.axis("off")
#plt.show()

#不同关键词word对应的sales之和的统计分析
#说明：例如 词语 ‘简约’，则统计商品标题中含有‘简约’一词的商品的销量之和，即求出具有‘简约’风格的商品销量之和
import numpy as np
w_s_sum = []
for w in word_count.word:
    i = 0
    s_list=[]
    for t in title_clean_dist:
        if w in t:
            try:
                s_list.append(data.sales[i])
            except:
                s_list.append(0)

        i += 1
    w_s_sum.append(sum(s_list))

df_w_s_sum = pd.DataFrame({'w_s_sum':w_s_sum})
print(word_count.head())
print(df_w_s_sum.head())

df_word_sum = pd.concat([word_count,df_w_s_sum],axis=1,ignore_index = True)
df_word_sum.columns = ['word','count','w_s_sum']
print(df_word_sum.head(30))

# 可视化数据
df_word_sum.sort_values('w_s_sum',inplace=True,ascending=True)
df_w_s = df_word_sum.tail(30)

import  matplotlib
from matplotlib import pyplot as plt
# matplotlib默认不支持中文的，所以一定要用自定义字体，下面绘图标题/行/列都用的到
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
font = matplotlib.font_manager.FontProperties(fname='./data/DroidSansFallbackFull.ttf')
index = np.arange(df_w_s.word.size)

plt.figure(figsize=(6,12))
plt.barh(index,
         df_w_s.w_s_sum,
         color='blue',
         align='center',
         alpha=0.8)
plt.yticks(index, list(df_w_s.word), fontproperties=font)
for y,x in zip(index, df_w_s.w_s_sum):
    plt.text(x,y,"%.0f" %x, ha='left', va = 'center')

#plt.show()

#商品的价格分布情况分析

data_p = data[data['view_price'] < 20000]

plt.figure(figsize=(7,5))
plt.hist(data_p['view_price'], bins=15, color='purple')
plt.xlabel(u"价格",fontsize=12,fontproperties=font)
plt.ylabel(u"商品数量",fontsize=12,fontproperties=font)
plt.title(u"不同价格对应的商品数量分布",fontsize=15,fontproperties=font)
#plt.show()

#不同价格区间的商品的平均销量分布：
data['price'] = data.view_price.astype('int')
data['group'] = pd.qcut(data.price,12)#将price列分为12组
dr_group = data.group.value_counts().reset_index()
#以group列进行分类求sales的均值
df_s_g = data[['sales','group']].groupby('group').mean().reset_index()

#柱型图
index = np.arange(df_s_g.group.size)
plt.figure(figsize=(8,4))
plt.bar(index,df_s_g.sales,color='purple')
plt.xticks(index,df_s_g.group,fontsize=12,rotation=30)
plt.xlabel('group')
plt.ylabel('mean_sales')
plt.title('不同价格区间的商品平均销量',fontproperties=font)
#plt.show()

# 商品价格对销量的影响分析
fig, ax = plt.subplots()
ax.scatter(data_p['view_price'],data_p['sales'],color='purple')
ax.set_xlabel('价格',fontproperties=font)
ax.set_ylabel('销量',fontproperties=font)
ax.set_title('商品价格对销量的影响',fontproperties=font)
#plt.show()

#商品价格对销售额的影响分析
data['GMV'] = data['price'] * data['sales']
import seaborn as sns
sns.regplot(x="price",y='GMV',data=data,color='purple')
#plt.show()

# 不同省份的商品数量分布
plt.figure(figsize=(8,4))
data.province.value_counts().plot(kind='bar',color='purple')
plt.xticks(rotation=0)
plt.xlabel('省份',fontproperties=font)
plt.ylabel('数量',fontproperties=font)
plt.title('不同省份的商品数量分布',fontproperties=font)
#plt.show()

# 不同省份的商品平均销量分布
pro_sales = data.pivot_table(index = 'province',values = 'sales',aggfunc=np.mean)# 分类求均值
pro_sales.sort_values('sales',inplace=True,ascending=False)# 排序
pro_sales = pro_sales.reset_index()
index = np.arange(pro_sales.sales.size)
plt.figure(figsize=(8,6))
plt.bar(index,pro_sales.sales,color='purple')
plt.xticks(index,pro_sales.province,fontsize=11,rotation=0,fontproperties=font)
plt.xlabel('province',fontproperties=font)
plt.ylabel('mean_sales',fontproperties=font)
plt.title('不同省份的商品平均销量分布',fontproperties=font)
plt.show()

#导出数据绘制热力型地图
#pro_sales.to_excel('./data/pro_sales.xlsx',index=False)
#可以使用百度提供的API绘制热力图，本人很懒，没去注册所以这块就不写了