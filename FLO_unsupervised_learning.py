###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering )  müşteriler kümelere ayrılıp ve davranışları gözlemlenmek istenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# 20.000 gözlem, 13 değişken

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type : 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.


###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv.csv verisini okuyunuz.
           # 2. Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz. Tenure(Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# GÖREV 2: K-Means ile Müşteri Segmentasyonu
           # 1. Değişkenleri standartlaştırınız.
           # 2. Optimum küme sayısını belirleyiniz.
           # 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
           # 4. Herbir segmenti istatistiksel olarak inceleyeniz.

# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
           # 1. Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
           # 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
           # 3. Herbir segmenti istatistiksel olarak inceleyeniz.


###############################################################
# GÖREV 1: Veri setini okutunuz ve müşterileri segmentlerken kullanıcağınız değişkenleri seçiniz.
###############################################################

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)




def load_csv_to_df(filename):
    current_directory = os.getcwd()
    full_path = os.path.join(current_directory, filename)
    df = pd.read_csv(full_path)
    return df

df_ = load_csv_to_df('flo_data_20k.csv')

df = df_.copy()

# Tarih değişkenine çevirme
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() # 2021-05-30
analysis_date = df["last_order_date"].max() + pd.Timedelta(days=2)
# analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).dt.days
df["tenure"] = (df["last_order_date"] - df["first_order_date"]).dt.days

###############################################################
# Kullanılacak Değişkenleri Seçme

df.columns
# Birden fazla master_id var mı? Hayır
df['master_id'].value_counts().max()

# Tek seferlik alışverişleri mi yansıtıyor? Yoksa Arka tarafta işlemler mi yapılıyor?
df.order_channel.value_counts()
df.last_order_channel.value_counts()

# En eski müşteri kim?
df[df['first_order_date']==df.first_order_date.min()]

df[df['last_order_channel'] == 'Offline'][['order_channel','last_order_channel']]

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]

model_df.head()
###############################################################




###############################################################
# GÖREV 2: K-Means ile Müşteri Segmentasyonu
###############################################################

# 1. Değişkenleri standartlaştırınız.
#SKEWNESS
def check_skew(df_skew, column, useLog=False):
    try:
        skew = stats.skew(df_skew[column])
        skewtest = stats.skewtest(df_skew[column])
        plt.title('Distribution of ' + column)
        
        # Checking if log scale is requested and data is strictly positive
        if useLog and (df_skew[column] > 0).all():
            sns.histplot(np.log(df_skew[column]), kde=True, color="g")
        elif useLog and not (df_skew[column] > 0).all():
            print("Data contains zero or negative values, cannot use logarithmic scale")
            sns.histplot(df_skew[column], kde=True, color="g")
        else:
            sns.histplot(df_skew[column], kde=True, color="g")
            
        print("{}'s Skew: {}, Skew Test Result: {}".format(column, skew, skewtest))
    except Exception as e:
        print(f"Error processing column {column}: {e}")


plt.figure(figsize=(9, 9))

plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')

plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')

plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')

plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')

plt.subplot(6, 1, 5)
check_skew(model_df,'recency')

plt.tight_layout() 
plt.show()  


# Normal dağılımın sağlanması için Log transformation uygulanması
def apply_log_transform(df, columns):
    """
    Apply log transformation (log1p) to specified columns of a DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the columns to transform.
    columns (list of str): List of column names to transform.

    Returns:
    pd.DataFrame: DataFrame with the log transformations applied.
    """
    for column in columns:
        # Check if the column exists in the DataFrame
        if column in df.columns:
            # Apply np.log1p which is log(x + 1) to handle zero values safely
            df[column] = np.log1p(df[column])
        else:
            print(f"Column '{column}' not found in DataFrame.")
    return df

model_df = apply_log_transform(model_df, model_df.columns)

model_df.head()


# Scaling
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()


# 2. Optimum küme sayısını belirleyiniz.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()




# k-elbow değerini kullanarak optimum küme sayısını belirleme
def calculate_silhouette_scores(data, start_k, end_k, random_state=42):
    """
    Calculate K-Means clustering silhouette scores for a range of k values.

    Args:
    data (pd.DataFrame): The dataset to cluster.
    start_k (int): The starting number of clusters.
    end_k (int): The ending number of clusters (inclusive).
    random_state (int): A seed value to ensure reproducibility.

    Returns:
    dict: A dictionary of k values and their corresponding silhouette scores.
    """
    silhouette_scores = {}
    for k in range(start_k, end_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores[k] = score
        print(f"Küme sayısı: {k}, Silhouette Skoru: {score:.2f}")
    
    return silhouette_scores

# Veri setiniz üzerinde 2'den 20'ye kadar silhouette skorlarını hesapla
silhouette_scores = calculate_silhouette_scores(model_df, 2, 20)



# 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments=k_means.labels_
segments

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()


# 4. Herbir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency":["mean","min","max"],
                                  "tenure":["mean","min","max","count"]})


###############################################################
# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
###############################################################

# 1. Görev 2'de standarlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show()


# 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()

# 3. Herbir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency":["mean","min","max"],
                                  "tenure":["mean","min","max","count"]})

