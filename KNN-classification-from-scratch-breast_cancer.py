import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
data = load_breast_cancer()
X = data.data
y = data.target
colors = ['red','blue']
y_colors = [colors[label] for label in y]    
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)    
new_point = X[34]
new_point_scaled = scaler.transform([new_point])[0]

def euclidean_distance(p,q):
    return np.sqrt(np.sum((p-q)**2))
class KNN:
    def __init__(self,k):
        self.k = k

    def fit(self,X,y):
        self.X = X
        self.y = y
    def predict(self,new_point):
        distances = []
        for i in range(len(self.X)):
            distance = euclidean_distance(self.X[i],new_point)
            distances.append([distance,self.y[i]])
        distances.sort()
        nearest_neighbors = distances[:self.k]
        
        labels = [label for _,label in nearest_neighbors]
        result = Counter(labels).most_common(1)[0][0]
        return result


clf = KNN(k = 5)
clf.fit(x_scaled,y_colors)
predicted_color = clf.predict(new_point_scaled)
print(predicted_color)


pca = PCA(n_components =3)
X_vis =pca.fit_transform(x_scaled)
new_point_vis = X_vis[0]
       
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.grid(True)
ax.scatter(new_point_vis[0],new_point_vis[1],new_point_vis[2],marker='*',color=predicted_color)
for i, point in enumerate(X_vis):
    ax.scatter(point[0],point[1],point[2],color=y_colors[i],alpha=0.6)
for i,point in enumerate(X_vis):
    dist = euclidean_distance(x_scaled[i],new_point_scaled)
    ax.plot(
    [new_point_vis[0],point[0]],
    [new_point_vis[1],point[1]],
    [new_point_vis[2],point[2]],
    color='black',alpha=0.5,
    linewidth=1,linestyle=':')
    mid = (new_point_vis+point)/2
    ax.text(mid[0],mid[1],mid[2],f"{dist:.2f}")
plt.title("KNN  Visualization (PCA Reduced)")
plt.show()
