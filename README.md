# Customer Segmentation with Unsupervised Learning (FLO)

## 📌 Project Overview
In this project, customer segmentation was performed using **unsupervised learning techniques** to better understand customer behaviors and purchasing patterns.  
Both **K-Means** and **Hierarchical Clustering** methods were applied, and resulting segments were analyzed from a business perspective.

---

## 🧠 Business Problem
The goal is to segment customers based on their historical omnichannel purchasing behavior and analyze these segments to support data-driven marketing and customer strategy decisions.

---

## 📊 Dataset Information
- Customers who made purchases between **2020–2021**
- Omnichannel data (online & offline)
- **20,000 observations, 13 variables**

Key variables include:
- Total online/offline order counts  
- Total online/offline spending  
- Recency (days since last purchase)  
- Tenure (customer lifetime in days)  

---

## ⚙️ Methodology

### 1️⃣ Data Preparation & Feature Engineering
- Date variables converted to datetime format  
- New features created:
  - **Recency**: Days since last purchase
  - **Tenure**: Customer lifetime  
- Behavioral and monetary variables selected  
- Log transformation applied to reduce skewness  
- Features scaled using **MinMaxScaler**

### 2️⃣ K-Means Clustering
- Optimal number of clusters determined using:
  - Elbow Method
  - Silhouette Score  
- Customers segmented using K-Means  
- Segment-level statistical analysis performed  

### 3️⃣ Hierarchical Clustering
- Dendrogram used to determine optimal cluster count  
- Agglomerative Clustering applied  
- Segment behaviors analyzed statistically  

---

## 📈 Key Outcomes
- Identified distinct customer segments based on purchasing behavior  
- Compared K-Means and Hierarchical Clustering approaches  
- Generated actionable insights for customer-focused strategies  

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- SciPy  
- Matplotlib, Seaborn  
- Yellowbrick  
