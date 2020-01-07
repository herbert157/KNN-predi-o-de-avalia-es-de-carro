import pandas as pd

base = pd.read_csv('car.data')

previsores =  base.iloc[:,0:6].values
classe = base.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder

labelEnconder = LabelEncoder()

previsores[:,0] = labelEnconder.fit_transform(previsores[:,0])
previsores[:,1] = labelEnconder.fit_transform(previsores[:,1])
previsores[:,2] = labelEnconder.fit_transform(previsores[:,2])
previsores[:,3] = labelEnconder.fit_transform(previsores[:,3])
previsores[:,4] = labelEnconder.fit_transform(previsores[:,4])
previsores[:,5] = labelEnconder.fit_transform(previsores[:,5])

from sklearn.preprocessing import StandardScaler

escalonamento = StandardScaler()

previsores = escalonamento.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(classe_teste)