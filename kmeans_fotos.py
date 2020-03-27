import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster


archivos = glob.glob('Imagenes/*.png')

imagenes = []
for archivo in archivos:
    imagen = plt.imread(archivo)
    imagenes.append(imagen.reshape(30000))
    
    
X = np.array(imagenes)

inercias = []
N_max = 21
for N in range(1,N_max):
    k_means = sklearn.cluster.KMeans(n_clusters=N)
    k_means.fit(X)
    inercias.append(k_means.inertia_)
    

plt.figure()
plt.plot(range(1,N_max),inercias)
plt.ylabel('Inercia')
plt.xlabel('Número de clusters')
plt.title('Inercia')
plt.xticks(np.arange(2,N_max,2))
plt.savefig('inercia.png')

N_best = 4
k_means = sklearn.cluster.KMeans(n_clusters=N_best)
k_means.fit(X)
centros = k_means.cluster_centers_
distancias = np.zeros((N_best,len(X)))


for i in range(len(X)):
    for j in range(N_best):
        distancias[j,i] = np.sqrt(np.sum(np.square(X[i]-centros[j])))

sorts = []
for j in range(N_best):
    sorts.append(np.argsort(distancias[j]))
    
    
fig,axes = plt.subplots(N_best,5,figsize=(20,N_best*5))
print(np.shape(axes))

for j in range(N_best):
    for i,ax in enumerate(axes[j]):
        ax.imshow(X[sorts[j]][i].reshape((100,100,3)))
        ax.set_title('Distancia = {:.2f}'.format(distancias[j][sorts[j]][i]))
    axes[j][0].set_ylabel('K='+str(j+1),fontsize=20)

fig.savefig('ejemplo_clases.png')