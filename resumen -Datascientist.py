#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###Análisis exploratorio
    - Descripción de datos: maximos, minimos, medias, mediana (escalado, el valor del index medio), moda (mayor frecuencia), percentiles, distribuciónes
    - Qty de nulos,duplicados, unique values
    - Distribución de la variable a predecir (filterr permite identificar a que distribución se asemeja. abajo teoria de distribuciones)
    - Variables cuantitativas con valores concentrados (conviene pasarlo a cuantiativo)
    - Identificación de outliers (boxplot por ejemplo sirve, percentiles,etc.)
    - Correlación de variables (https://towardsdatascience.com/an-interactive-guide-to-hypothesis-testing-in-python-979f4d62d85):
        - Vcategorica vs V categorica: chi cuadrado test. sns.snsbarplot
        - Vnumerica vs Vcategorica:  T Testn(menor a 2 grupos) y ANOVA test (mayor a 2 grupos)
        -Vnumerica vs Vnumerica: Coeficiente pearson. -va de –1 (relación negativa) y +1(relación positiva. cercano a 0 no hay correlación.Mide solo la relación con una línea recta. Para visualizarlo: scatter plot. Limpiar outliers. Analiza covarianza vs varianzas
    - Gráficos (según sean: Cuanti, Cuali o time series): histogramas, barplot, line plot, boxplot, violin plot,heatmap, scatters. Evolución en el tiempo
    - Aprendizaje desbalanceado? -30% de una clasificación 
        - Ajuste de Parámetros del modelo: l parámetro class_weight= “balanced”. redes neurnales ajustando loss
        - Muestras artificiales: con RandomOverSampler
        - Modificar el Dataset: con  NearMiss
        - Balanced Ensemble Methods: Clasificador de Ensamble que utiliza Bagging
          https://www.aprendemachinelearning.com/clasificacion-con-datos-desbalanceados/

###Tratamiento de datos (data wrangling)
    - feature engineering (generación de nuevas features o selección)
        - Feature importance (Existe tambien LIME, que para cada caso particular, indica cual vue el feature que más ponderó para clasificar/regresionar) LIME (local model interpretability ) . Explica clasificaciones de cualquier modelo de ML, como? Inicia generando leves perturbaciones o eliminando features del input que se quiere explicar, y enviándolo de nuevo al modelo. Con todos los resultados, se lo envía a un modelo interpretador como decision tree y analizando los resultados. Asi, los features con mayor frecuencia que al modificarse marginalmente generaron un cambio en la clasificación, serán las variables más ponderantes para ese caso.  
        - Reducción de dimensionalidad (siempre conviene lo minimo e indispensable. A veces como un embedding): reducen los features entendiendo que no se pierde información y disminuyendo el ruido y la posibilidad de overfitting
                - k-means
                - algoritmos jerárquicos (van a su nivel superior)
                - Principal Component Analysis (PCA)
                - T-distributed Stochastic Neighbor Embedding (t-SNE)
                - SVD 
        - Seleccion de features según correlaciones
        - Generación de nuevos features
    - Como trabajar los nulos, duplicados y outliers
    - Categorical features to numerical vector format (pd.get_dummies/OneHotEncoder/LabelEncoder). Category encoder es la libreria que conviene usar! onehotencoder, no dummy.
    - Standarizar si tienen tamaños grandes y variados
    - ajustar formatos (castear)
    - data leakage(fuga): revisar que no haya features que no se esperaría que estuviera disponible en el momento de la predicción


### optimizador de parametros:
    - Grid Search
    - Random Search
    - basados en gradiente

### Modelos: Predicción,Asociación, Clasificación, Clustering, patrones
    - Libreria tpot es una sirve para elegir el mejor modelo, con los parametros relativamente bueno.
    - Classification Models -Sup- (devuelve valor discrto, categoría): Logistic regresion (solo binaria), KNN,Naive bayes,SVM, Decision tree(xboost entra aca), random forest
    - Regression Models -Sup- (devuelve valor continuo): linear regression, lasso Regresion, Ridge regression, SVM  regressor, decision tree regressor,  etc.
    - Clustering -Unsup-(agrupa elemenos por densidad (grupos x concentración), distribución (grupos por concentración. hay que conocer la distribución de los datos), centroides (por cercanía a centroides aleatoreos. itera sobre todos los puntos) o jerarquia): 
        k-means,
        K means – Simple,
        K means++,K medoids, 
        Agglomerative clustering,
        DBSCAN. Si descarta puntos
        Afinity propagation. No descarta ningun valor pero se ubica en lugaresde alta densidad. devuelve centros de la muestra. Es en formas esfericas
        BIRCH
        Algoritmo de agrupamiento por Propagación de Afinidad (make_classification)
    - Dimensionality Reduction -Unsup- (Los uso en tratamiento de datos arriba)
    - Deep Learning -Reinf- etc.                                                                                          
    - Time series: ARIMA,ARCH,GARCH, Facebook Prophet
    - Clasificación con Naive Bayes:Considera que cada una de las features contribuye de manera independiente a la probabilidad predictiva.
        Cada variable independiente aumenta o disminuye la probabilidad a que se predija que es un hombre (si tengo altura, peso y talle)
    - Ensambles: 
    - Otros: Cadena de markov
    - Redes
        - learning_rate, relacionado a optimazer
        - loss: muy parecido al error que voy a medir 
        - optimzer (velocidad de corrección)
        - epochs: qty de iteraciones sobre el train
        - batch: qty de muestras con la que se updatea el error del modelo en el entrenamiento. Mientras más grande el batch, más lento entrena por acumular datos temporales, pero se vuelve más preciso porque ajusta los parametros en base a muestras grandes. Si entrenamos con batch chicos, va a ser mas rapido, pero con menor precisión porque cada ajuste se hace según una muestra chica
        - accuracy: aciertos sobre intentos

### Entrenamiento

### UTILIZACIÓN DE LIME PARA IDENTIFICAR LOS FEATURES PONDERANTES DE 1 SAMPLE
Resultados & Evaluación & Métricas
    - underfitting (entrenamiento pobre. tengo high bias y low variance) y overfitting (me pase de mambo. tengo low bias pero high variance). Cuando el error de la validación es minimo, es el punto optimo
        - precisión/low variance (dispersión): En el entrenamiento, todas las muestras dan al mismo punto. 
        - exactitud/low bias (posición media): En el entrenamiento, el valor medio coincide con el verdadero valor de la magnitud medida
        - La validación tiene alta precisión/low variance y baja exactitud/low bias: Overfitting. Muchos epochs, valores duplicados en entrenamiento
        - La validación tiene alta exactitud/low bias pero baja precisión/low variance: Underfitting. X ruido de entrenamiento, se entreno poco, muestra chica, 
    - metricas: MAE,MSE,RMSE,R2,R2 ajustado 
    - Confusion matrix: calcular Precision, Recall, F1, Accuracy 
    - Curvas  ROC (falsos negativos, etc.)
    - validación cruzada (cross_val_score): busca garantizar que los resultados son independientes de la partición train/test. Lo que hace: Particiona de forma diferente en train y test, digamos 6 veces y despues compara los resultados
                                                                                         
                                                                                         

Distribuciones:
    CONTINUAS
        - Uniforme
        - Normal
        - Gamma
        - chi cuadrado (caso especial de gamma)
        - Pareto
        - t student
                                                                                         
    DISCRETAS
        - Uniforme (1 dado)
        - bernulli (binario, con una proba)
        - Geometrica.La probabilidad de tener un exito en tantos intentos
        - Pascal/Distribución binomial negativa. Cantidad de dias transcurridos 4 fallas de maquina
        - Poisson (Se especializa en la probabilidad de ocurrencia de sucesos con probabilidades muy pequeñas en el tiempo. Ej: falla en maquinas)


# - SUPERVISADO
#     - REGRESION Devuelve un valor como predicción (time series es un tipo, donde las variables dependientes son temporales)
#          Regresión lineal (Ride o Lasso), Regresión polinomeal, SVM (Support vector machine) - regresión , decisiontreeregressor,randomforestregressor, Redes neuronales.
#          algunos para time series: ForecasterAutoreg,ForecasterAutoregCustom,ForecasterAutoregMultiOutput,LSTM,MonteCarlo
# 
#     - CLASIFICACIÓN Devuelve una categoria 
#         Logistic regression, Naive Baye, K-NN (k-nearest)/Vecinos más cercanos, SVM (Support vector machine),  Descision trees - clasificación, random forest - clasificación,Neuronal Networks..
#         
#         
# - NO SUPERVISADO
#     - CLUSTERING: Exclusivo, Aglomerativo, Solapamiento, Probabilístico 
# Divide data by similar features. Usa para: tipo de cliente, imagenes, points in maps (Clients in high or low)
#         . Allgorithms: K-means, Agglomeratime, Mean-shiff, fuzzy c-means, 
#     - PATTERN SEARCH/ Search relations/ ASOCIACION. used in goods bought together, place products in shelves (If some buy a laptom, seguramente wants a mouse)
#          .allgorithms:Euclat, Apriori, FP-growth
# 
#    -DIMENSION REDUCTION. used for Recomend systems, risk managments, text. Create grupos by more relevant features (Netflix, identificar principales palabras de texto y los agrupa en : textos educativos, entretenimineto, etc.)
#         . Algorithsms: PCA, SVD, LDA
# 
# 
# - REFUERZO.There is no data, but yes an enviroment (Ej: playing chess with the roules, games, snake, self-driving cars, vacuums robots
#         .Allgorithsms: Genetic Algorithm, SARSA, A3C, Qlearning, Deep Q-network
#          Redes neuronales 
#          
#          
# ENSAMBLE (pipline): Use as many allgorithms as possible and choose the average solution. 
#         .Usually in decision trees in forest, or using many algorithms (Any)

#     NLP
#         Clasificar texto: Beto (Auto-Encoder)
#         predicción de palabras: Auto-Decoder 
#         Resumir texto: Encoder-Decoder 
#         identificar categorías: LDA (Latent Dirichlet Allocation) unsupervised
#         Traducción:Encoder-Decoder
#         chatbot: Encoder-Decoder
#         Reconocimiento de voz: ? investigar wev2vec2?
# 
# ################################# NLP especialidad
# 
# #########sirve para: chatbots, resumenes, correción, predicción de palabra, generación de texto,clasificación de texto,
# 
# ########Arquitecturas de redes neurnales:
# RNN - recurrent neuronal networks
# LSTM - Long short term memory
# TRANSFORMERS. algoritmos pretrain a ser fine-tune: BERT (encoder)/BETO(encoder)/GPT3(decoder)
# 
# 
# #################Procesos NLP:
# Limpieza (stopwrods, lower, simbols,special characters etc.)
# Tokenización (cada palabra pasa a ser una unidad. la palabra es la unidad minima),
# Lemmatización/stemmatización (busca la relación entre las palabras y busca su raiz madre. ya sea por sufijos/prefijos o mismo verbo por ejemplo. Correr,corren, corri)
# Sequencing (cada palabra pasa a ser un numero)
# Padding,(para el armado de frases, define la dimensión de un vector, tal que si hay menos palabras, pone ceros)
# POS- tagging part of speach (estructura en verbos, sustantivos, etc.),
# Chunking - Process of extracting phrases (chunks) from unstructured text. Identifica frases, que las palabras por si solas significan otra cosa! ejemplo, check out, run out,etc.
# NER (name entity recognition. reconoce persona, entidad, empresa, localidad)
# embedding (cada palabra pasa a ser un vector de X dimensión según la cantidad de palabras. Una red neurnal que tiene de input la sequencia de tokens, y devuelve un vector de 6 dimensiones por ejemplo), 
# DISTANCIAS ENTRE PUNTOS UTILES: cosine_pearson : 0.8280372842978689, cosine_spearman : 0.8232689765056079, euclidean_pearson : 0.81021993884437, euclidean_spearman : 0.8087904592393836, manhattan_pearson : 0.809645390126291, manhattan_spearman : 0.8077035464970413, dot_pearson : 0.7803662255836028, dot_spearman : 0.7699607641618339
# positional encoder (solo para transformers)
# 
# Librerias NLP:
#     para redes: tensorflow, keras
#     ejercicios de NLP: NLTK,pytroch(facebook), transformers, SpaCy, TextBlob, nlp (Libreria de huggingface de Datasets)
#     
#  ###### Comunidades NLP
# dotcsv
# codificando bits
# towardsdatacience
# kaggle
# huggingface
# medium
# nlp en es (huggingface)
# 

# ## Exploración

# In[98]:





# In[99]:


import pandas as pd
import numpy as np
df=pd.DataFrame({'feature1':range(0,20),'feature2':np.arange(0, 40, 2),'feature3':np.linspace(0, 2, 20),'feature4':np.zeros(20),'feature5':None,'label':['a','b','b','b','a','c','b','b','b','c','b','c','a','b','d','d','d','a','c','d']})


# In[100]:


df


# In[102]:


#transforma a numeros los labels
from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
df['10'] = le.fit_transform(df['feature1'])                    


# In[103]:


len(df['10'].unique())


# In[ ]:


a tener en cuenta:
    
df.iloc[68,1:-12]
df.loc[]
'# nulls in train: {variable1}'.format(df.isnull().sum().sum()


# In[ ]:


df.shape


# In[ ]:


df.info()
# df.dtypes


# In[29]:


df.describe()


# In[44]:


#qty valores unicos por columna
df.nunique()


# In[39]:


df['variable1'].value_counts()

df['variable1'].unique()
# In[ ]:


df.isna()


# In[41]:


#qty valores nulos por columna
df.isna().sum().sort_values()


# In[42]:


#qty valores duplicados por columna
df.duplicated().sum()


# In[ ]:


#Tablas pivots o group by 
df[['label','feature1']].groupby('label').count().sort_values('feature1',ascending=False)

df.pivot_table(index=['Year', 'Semana', 'Site'], values=['GB', 'Fee'], aggfunc=np.sum).reset_index().sort_values(by='GB', ascending=True) # si agrego .reset_index me vuelve a la base plana!


# In[ ]:


#A que distribución se asemeja una variable?

distribuciones = ['cauchy', 'chi2', 'expon',  'exponpow', 'gamma',
                  'norm', 'powerlaw', 'beta', 'logistic']

fitter = Fitter(df['varialbe1'], distributions=distribuciones)
fitter.fit()
fitter.summary(Nbest=10, plot=False)


# In[ ]:


# Todas las modificaciones se realizan sobre el ultimo grafico

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import style
from matplotlib.figure import Figure
import matplotlib as mpl
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'notebook #Hace que sea interactivo')
# toda la clasificacion de graficos! en https://www.data-to-viz.com/ y con el codigo

plt.figure(figsize=[16,4]) # create a new figure. CREATE. If note, you will modify last figure
plt.clf() # Start over

TIPOS DE GRAFICOS
plt.subplots(5,2,8, sharex=True, sharey=True) #(2 columnas), (3 filas), s mi 8vo graph). lo que plotee ahora estara en esta posiciónplt.plot(3, 2, '.') #puntos 

# muchos puntos
plt.scatter(Array,Array, s=10, c='red', label='Tall students') 

plt.stackplot([2,3,4],[6,3,8],[9,10,4],colors=['g','m'])
plt.plot(Array1, '-o',Array2, '-o') #Linea
plt.plot([22,44,55], '--r') #Linea punteada

plt.bar([1,3,5,7,9],[5,13,8,2,9], width = 0.3) #barras

#Histograma
plt.hist(df['gamma'], bins=100) #Histograma

plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range') # Box and Whisker Plots. 'range' cause dont show outliers

sns.countplot(x='label',data=df)

g.set_xticklabels(g.get_xticklabels(),rotation=90);plt.hist2d(Array,Array, bins=100) #Heatmap

pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap) # analiza correlaciones individuales de un grupo de features

sns.violinplot( x= colum, y= 'precio',data = datos,color = "white",ax    = axes[i])

FORMATOS
plt.xlabel('TITLE') # add a label to the x axis
plt.ylabel('aXIS y') # add a label to the y axis
plt.title('Axis X') # add a title
matplotlib.rcParams.update({'font.size': 16}) #modificas tamaño de letra
plt.pie
plt.legend(loc=4, frameon=False, title='Legend') # add a legend (uses the labels from plt.scatter) loc=(1,0) tambien sirve!
plt.subplots_adjust(bottom=0.25) # adjust the subplot so the text doesn't run off the image
plt.grid #Fondo rayado
plt.colorbar() #Add a colorbar legend
plt.xticks((pivot30.index), ('V1', 'V2', 'V3', 'V4', 'V5')) #para nombrar a las variables del eje X
plt.set_xticklabels(g.get_xticklabels(),rotation=90);

ESTILO
style.use('ggplot') #por ejemplo. 
plt.style.available para ver todos los estilos

EJES ???
ax = plt.gca() # get current axes. Los ejes ahora se llaman ax
ax.axis([0,6,1,10]) # Set axis properties [xmin, xmax, ymin, ymax]
plt.gca().get_children()


#Funcion lineal
plt.subplots(figsize = (12,6))
fig=data.set_index("Date")["price_use"].plot()
fig.set(xlabel="", ylabel = "USD/MMBTU", title="Peruvian Exported Gas Price")
plt.show()


#Plotting distribution of markers by royalties
f, ax = plt.subplots(figsize = (12,6))
fig=sns.violinplot(x="marker", y="%regalia", data=data)
fig.set_title("Distribution: Royalty by Marker")
plt.show(
    
rolling(10).mean() #grafica la media movil! de los 10 ultimos valores por ejemplo


# In[ ]:


#Pearson correlation (Analiza correlación de variables cuantitativas)
# pearson_corr = pd.DataFrame(train_featured.corrwith(train_featured['congestion'], method='pearson'), 
#                             columns=['congestion'])

#gráfico correlación de features
import seaborn as sb
sb.pairplot(df.dropna(), hue='label',size=4,vars=['feature1','feature2'],kind='scatter')


# Heatmap matriz de correlaciones
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

sns.heatmap(
     df.select_dtypes(include=['float64', 'int']).corr(method='pearson'),
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 6},
    vmin      = -1,
    vmax      = 1,
    center    = 0,
    cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True,
    ax        = ax
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
)

ax.tick_params(labelsize = 8)


# # TRATAMIENTO DE DATOS
# 

# In[31]:


# df['feature1'].split(1,expand=Truee)
df.replace({1:'uno ,dos',3:'tres,tee'})
df = df['feature1','feature2'] #renombro columnas según el orden


#Seleccion de columnas
df = df[['feature1','feature2']]
#Cambio nobmres columna
df.columns = ['variable1','variable2']


# In[ ]:


#Valores nulos
df.dropna()

#Valores duplicados


# In[107]:


# Tratamiento features categoricos

# Category Encoders es la libreria. Dummy queda obsoleto, porque por ejemplo, si tengo test y train, y en test no hay una categoría. esto rompe con dummy, no con onehotencoder (o otras funciones de category encoder)


#transforma a numeros los labels
le = LabelEncoder()
df['categ'] = le.fit_transform(df.label)


# Los features categóricos fueron sometidos a distintos tipos de “encoding” (JamesStein, OneHotEncoder, HashingEncoder, HelmertEnconder, etc.).
#Dummys tambien sirve

from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Convert the 10 bacteria names to the integers 0 .. 9
le = LabelEncoder()
df_train['category'] = le.fit_transform(df_train.label)
# dummy


# In[ ]:


# Standalizar el input si hay valores de escalas muy diferentes

#Opcion 1
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler().fit(df_train[features])

# df_train[features] = scaler.transform(df_train[features])
# df_test[features]= scaler.transform(df_test[features])



#Otra opción para standarizar
# A veces, para standarizar la información y que todas las features tengan misma importancia 
# (sino a veces una por la escala tiene mayor o menor peso) se hace lo siguiente. todas varian entre 0 y 1


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
                                                   random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)


# Exploración con modelos de kmeans (no supervisado)

# In[ ]:


#### KMEANS
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

df_km=df.copy()
X_km=df_km.drop(['label'],1)
Y_km=df_km[['label']]

#Matodo1
# from sklearn.metrics import silhouette_score
# range_n_clusters=[2,3,4,5,6,7]
# for n_clusters in range_n_clusters:
#     clusterer = KMeans(n_clusters=n_clusters)
#     preds = clusterer.fit_predict(X_km)
#     centers = clusterer.cluster_centers_
#     score = silhouette_score(X_km, preds)
#     print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
    
#Maetodo2
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X_km).score(X_km) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[121]:


from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
kmeans = KMeans(n_clusters=3).fit(X_km)
centroids = kmeans.cluster_centers_
# print(centroids) #Me dice cuales son los puntos n_clusters que se usaran como centros

#Predigo cada fila del dataset según el kmeans
# kmeans.predict([X_train.loc[1].tolist()])
Y_groupkm=kmeans.predict(X_km)
score=kmeans.score(X_km)

#Creo un data frame para comparar las categorias que tengo con los que creo el kmeans
df_km['label_kmeans']=Y_groupkm
df_km['categ']=Y_km
df_km.reset_index(inplace=True)
df_km[['label_kmeans','categ','index']].groupby(['categ','label_kmeans']).count()


# Configuraciónes

# In[ ]:


Optimizador de parametros

#OPTIMIZADOR DE PARAMAMETROS DEL RANDOM FOREST. Nos va a decir, según las opciones que le demos, cual es la combinación más eficiente de:
# n_estimators,max_depth,min_samples_leaf

n_estimators=[200,350,500,700]
max_depth=[20,30,50,70]
min_samples_leaf=[2,5,10]
grid_param={'n_estimators':n_estimators, 'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf}

from sklearn.model_selection import RandomizedSearchCV
RFR=RandomForestRegressor(random_state=1)
RFR_random= RandomizedSearchCV(estimator= RFR, param_distributions= grid_param, n_iter=500, 
                               cv=5, verbose=2, random_state= 42, n_jobs=-1)

RFR_random.fit(X_trainb,Y_trainb)
print(RFR_random.best_params_)


# In[ ]:


# Tokenizar, Sequenciar y Paddleing

# from tensorflow.keras.preprocessing.text import Tokenizer
# tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>') #Si la palabra no esta entre las 10.000 mas usadas, se guarda como <UNK>
# tokenizer.fit_on_texts(tweets)

# tokenizer.texts_to_sequences([tweets[0]])
# lengh=[len(t.split(' ')) for t in tweets]

# sequences=tokenizer.texts_to_sequences(tweets)
# padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=maxlen) #Rellena con 0s para que todos los tweets tengan mismo largo


# In[140]:


#crear test,train y validation
from sklearn.model_selection import train_test_split
x_train, X_test, Y_train, Y_test = train_test_split(df.drop(['label'],1),df['label'],test_size=0.1)

#con validation
# X_train, X_test, Y_train, Y_test = train_test_split(df.drop(['label'],1), df['label'], train_size=0.8,random_state=42)
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, train_size=0.9,random_state=42)


# Modelos

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC #Suport vector machine (SVM)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.pipeline import Pipeline

forest
decisiontree
Gradient Boosting
SVM
KNN

#For reggresion
# extratreesRegressor() #El mejorcito! se lo usa en topes por ejemplo
#model= LinearRegression()
# model=Lasso(alpha=2.0, max_iter = 10000)
# model= Ridge(alpha=20.0)

#For clasification binaria
# model= LogisticRegression(C=100)


#For clasification
model = RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)
# model = ExtraTreesClassifier(n_estimators=100,n_jobs=-1,random_state=42)
# model = XGBClassifier(n_jobs=-1,n_estimators=1000,random_state=42)
# model = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
# model = MLPClassifier(random_state=1,max_iter=500,hidden_layer_sizes=[32,64,128,10],alpha=0.001) 
# model = SVC(kernel = 'linear', C=this_C) SVC(C=10) #Suport vector machine (SVM) clasification
# model = KNeighborsClassifier(n_neighbors = 5)
# algoritmo Gradient Boosting Regresor??
#  Extra Random Forest?

#For time series 1
# forecaster = ForecasterAutoreg(
#                 regressor = RandomForestRegressor(random_state=123),
#                 lags = 6
#              )

#For time series 2
# cat_base = CatBoostRegressor(
#     #ignored_features=ignore_cols,
#     cat_features=object_cols,
#     eval_metric='MAE'
# )


#Armado de piplines
# knn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
# model = Pipeline([
#     ('scale', StandardScaler()),
#     ('pca', PCA()),
# ('lda', LinearDiscriminantAnalysis(n_components=9)),
#     ('knn', knn)])




#############    REDES NEURONALES    #############

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

# - loss='sparse_categorical_crossentropy',  #Se define el error. Se define el objetivo, la función del error minimo que se quiere tener (hay infinitas de funciones de error minimo!!). Ejemplo:'sparse_categorical_crossentropy' que se usa para problemas de clasificacion
# Ejemplo, si tengo outlayers y tomo una el loss square (error al cuadrado), como estan mas lejos van a tener mayor peso y el algoritmo va a ponderarlos. en vez, si es loss absolute, no van a tener tanto peso a la hora de buscar ese error!

# - optimizer='adam', #Se encarga de ajustar la velocidad, tiempos, para llegar al punto más optimo, y tener la mayor accuracy (precisión). Ojo con overfitting aca! Si tengo poco dato conviene ir rapido, si tengo mucho dato, conviene ir lento. per si va muy lento, hay riesgo de overfitting y que sea imposible llegar al punto optimo.
# - metrics='accuracy'
# - epoch: cantidad de veces que entreno con mismos datos
# - bach: en cuanto divido el dataset para entrenar.
# - activation function: softmax, sigma, etc.

# Redes Neuronales
# Se puede optar por embedding o One Hot Encoding (que cada palabra sea un vector de 0s con 1 uno en la unique word)
# Embedding sirve para cualquier idioma! porque trabaja con tokens y sus relaciones en los inputs, no con texto

# Embedding: OBJETIVO, ES QUE LA RED LSTM CONSIDERE PALABRAS SEMEJANTES, COMO TALES
#lo que hace es generar 1 vector de 0s con solo 1 uno para cada unique word, y el output es el mismo vector, pero algunos 0s cambian de valor!! 
# En el entrenamiento, lo que se hace es que las mismas palabras tengan pesos semejantes segun contexto! Se entrena de forma supervisada, enseñandole que el output, es la frase.
# model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
# model.add(LSTM(hidden_size, return_sequences=True))


# import tensorflow as tf
# model = tf.keras.models.Sequential([
#         tf.keras.layers.Embedding(10000,16,input_length=maxlen), #maxlen es la dimensión del input!. 16 output interno. Cada palabra va a estar representado con un vector de 16 dimensiones
#         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=True)), #Bidirecional significa que el contexto puede provenir de la derecha o izq del texto. Son 2 LSTM pero en los 2 ordenes!!
#         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
#         tf.keras.layers.Dense(6,activation='softmax') #Output, 6 dimensiones (triste, happy, etc.)
# ])
# model.compile(
#     loss='sparse_categorical_crossentropy',  #Se define el error buscado. 'sparse_categorical_crossentropy' se usa para problemas de clasificacion. CrossEntropyLoss
#     optimizer='adam', #Se encarga de ajustar la velocidad, tiempos, para llegar al punto más optimo, y tener la mayor accuracy (precisión). Ojo con overfitting aca
#     metrics='accuracy'
# ) #loss y optimazer, usan el concepto de Gradient Descent (GD), que es cuan rapido quiero llegar al optimazer


# The example below defines a Sequential MLP model that accepts eight inputs,
# has one hidden layer with 10 nodes and then an output layer with one node 
# to predict a numerical value.
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense

#modelo con un input de 8, hidden de 10 y salida de 1 
# model = Sequential()
# model.add(Dense(10, input_shape=(8,)))
# model.add(Dense(1))

# #modelo con un input de 8, hidden de 100, hidden 80,hidden30 y salida de 1 
# model = Sequential()
# model.add(Dense(100, input_shape=(8,)))
# model.add(Dense(80)) #Crea una capa de 80 neuronas
# model.add(Dropout(0.2)) #en cada pasada de entrenamiento, aleatoreamente se eliminan 20% de la capa anterior, asi se evita overfitting!: Dropout has the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs.
# model.add(Dense(30))
# model.add(LSTM(30,return_sequences=True, batch_input_shape=(6,3,1))) #30 units (los features),return_sequences=True, devuelve misma dimensión de entrada para cada feature
# batch_input_shape=(6,3,1): significa que el batch es de 6, en cada unit entran 3 features,
# model.add(Dense(1, activation='sigmoid'))

# LSTM
# model.add(LSTM(30,return_sequences=True, batch_input_shape=(6,3,1))) #30 units (los features),return_sequences=True, devuelve misma dimensión de entrada para cada feature
# The output will have shape:
# (batch, arbitrary_steps, units) if return_sequences=True.
# (batch, units) if return_sequences=False.


# activation: sigmoid,relu,softmax(se activa la de valor máximo), tangh


# Entrenamiento

# In[209]:


# The batch size is a number of samples processed before the model is updated. 
# The number of epochs is the number of complete passes through the training dataset. repeticiones de mismo dataset

h=model.fit(X_train, y_train)

# h=model.fit(X_train,y_train, epochs=100, batch_size=32, verbose=2) #verbose:barrita de avance abajo
# h=model.fit(X_train, y_train, X_valid, y_valid,batch_size=20, n_epochs=20, learning_rate=0.05,random_state=42)

# h=model.fit(X_train, Y_train,validation_data=( X_valid, Y_valid),epochs=20,
#     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)]) #Si al segundo epoch el modelo no ve mejoras en accuracy, frena


# In[210]:


#History of loss and accuracy of epochs

get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nlp #Libreria de huggingface de Datasets
import random

epochs_trained = len(h.history['loss'])
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
plt.ylim([0., 1.])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[186]:


model.summary()


# Se analiza resultado & predicciones del modelo

# In[187]:


#Predict:
Y_pred=model.predict(X_test)
#Predict para NN en keras:
# pred_aux=model.predict(X_test)
# Y_pred=np.argmax(pred_aux, axis=1)


# In[211]:


#Analiza la relación de features en una clasificación (profundizar un poco)
X = X_train.values 
y = y_train.values 
cv_scores = cross_val_score(model, X, y)
print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'.format(np.mean(cv_scores)))


# In[192]:


SCORES A MEDIR, SUMADO CONFUSION MATRIZ

ERRORES:    

MAE (mean_absolute_error/error absoluto medio):El error promedio de las muestras. Es el más intuitivo y el que se usa para reportar. No el más util para trabajar
MSE (Mean Squared Error/Error cuadrático medio): Representa el error promedio por muestra al cuadrado. Mientras tenga más outliers o valores atípicos, mas castigará a este kpi. Efecto contrario si todos los errores son muy pequeños
RMSE (Root Mean Squared Error/Raiz Error cuadrático medio):Es la raiz de MSE. El error promedio x muestra. Como el MAE, pero castigado por outliers o errores groseros.
    
R2/coefficient of determination: 1-(MSE(vs valor real) / MSE(vs media)). MSE(vs media): es lo nuevo, y es el error más ingenuo y simplista, suponiendo que el modelo a predecir siempre será la media.
    Es una comparación de MSE vs el MSE de un modelo simplista. Da una idea de cuan bien estoy vs este MSE de referencia. si es cercano a 1, esta muy bien.
    Si R2 es negativo, conviene utilizar la media que el modelo para predecir (modelo malísimo).
    El problema es que si agrego variables, siempre va a acercarce a 1. No considera el ruido como un problema! sino como ayuda a disminuir el error.
R2 ajustado. Lo mismo que el R cuadrado, pero con una diferencia: El coeficiente de determinación ajustado penaliza la inclusión de variables.
MSPE – Error de porcentaje cuadrático medio
MAPE – Error porcentual absoluto medio
RMSLE – Error logarítmico cuadrático medio

# Metrics for regresions
model.coef_ # model coeff (w)
model.intercept_ #linear model intercept (b)
model.score(X_train, y_train) #'R-squared score (training)
model.score(X_test, y_test) #'R-squared score (test) El problema del coeficiente de determinación, y razón por el cual surge el coeficiente de determinación ajustado, radica en que no penaliza la inclusión de variables explicativas no significativas.

model.mean_absolute_error(X_train, y_train) #(MAE)


#Metrics forNN
# _=model.evaluate(X_test,Y_test)
# evaluate the model
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print('Test Accuracy: %.3f' % acc)


# In[ ]:


# Time Series
# la librería Skforecast dispone de la función grid_search_forecaster con la que comparar los resultados obtenidos con cada configuración del modelo.


# In[194]:


#feature importance 


# Option 0
#plot the catboost result
plot_feature_importance(model.get_feature_importance(), X_train.columns, 'CATBOOST')

#Option 1,En gral
from matplotlib import pyplot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

df_feature_importance=pd.DataFrame(model.feature_importances_,index=df_train.columns[1:-2].tolist())
df_feature_importance.sort_values(0,ascending=False).tail(50)

#Option2 si uso Xboost, este sirve para feature importance
# plot feature importance
# plot_importance(model)
# pyplot.show()

#option3,Plot feature importance 
# from adspy_shared_utilities import plot_feature_importances
# plt.figure(figsize=(10,4), dpi=80)
# plot_feature_importances(model, iris.feature_names)
# plt.show()
# print('Feature importances: {}'.format(model.feature_importances_))


# <!-- Normalizar un texto -->
# <!-- df.primary_diagnosis.value_counts(normalize = True).round(2) -->
# 

# In[195]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Opcion 1 de confusio
# cm = confusion_matrix(Y_test, Y_pred, normalize='true')
# plt.figure(figsize=(10, 10))
# sp = plt.subplot(1, 1, 1)
# ctx = sp.matshow(cm)
# plt.xticks(list(range(0, len(df_train.label.unique()))), labels=df_train.label.unique().tolist())
# plt.yticks(list(range(0, len(df_train.label.unique()))), labels=df_train.label.unique().tolist())
# plt.colorbar(ctx)
# plt.show()


#Opcion 2 de confusio
cm = confusion_matrix(Y_test,Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot() 
plt.show()


#Teoria
#CUANDO HAY SOLO 2 CATEGORIAS
# TP=cm[0][0]
# FN=cm[0][1]
# FP=cm[1][0]
# TN=cm[1][1]
# Sens=TP/(TP+FN) #Sensibilidad 
# UnomSp=FP/(FP+TN) #(1-Specify)
# Prec= TP/(TP+FP) # #Precisión
# # Lo mas eficiente es sensibilidad=1, 1-Sepcify=0
# # TP  FN  
# # FP  TN
#CUANDO HAY 3 CATEGORIAS
# TP1=cm[0][0]
# FN1=cm[0][1]+cm[0][2]
# FN1bis=cm[1][2]+cm[2][1]
# FP1=cm[1][0]+cm[2][0]
# TN1=cm[1][1]+cm[2][2]

# Sens=TP1/(TP1+FN1) #Sensibilidad 
# UnomSp=FP1/(FP1+TN1+FN1bis) #(1-Specify)
# Prec= TP1/(TP1+FP1) # #Precisión

# Lo mas eficiente es sensibilidad=1, 1-Sepcify=0
# TP  FN  
# FP  TN

# print(f'Resultados de {Y_test[0]}:')
# print(f'De los {TP1+FN1} comentarios test reales "{Y_test[0]}", el {round(Sens,2)*100}% fueron correctamente predictos como {Y_test[0]} (Sensibilidad)')
# print(f'De los {TN1+FP1+FN1bis} comentarios test reales "no {Y_test[0]}", el {round(UnomSp*100)}% fueron erroneamente predictos como {Y_test[0]} (1-Specify)') 
# print(f'De los {TP1+FP1} comentarios test predictos como "{Y_test[0]}", el {round(Prec*100)}% fueron correctamente predictos como {Y_test[0]} (Precisión/Accuracy)') 


# In[202]:


#Parecido a confusio

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall) 
# print('Accuracy: {:.2f}'.format(accuracy_score(Y_test, Y_pred)))
# print('Precision: {:.2f}'.format(precision_score(Y_test, Y_pred)))
# print('Recall: {:.2f}'.format(recall_score(Y_test, Y_pred)))
# print('F1: {:.2f}'.format(f1_score(Y_test, Y_pred)))


# Guardar modelo

# In[ ]:


pip install h5py
model.save('model.h5')


# In[ ]:


DATA ENGINEER


# In[ ]:


DATA ENGINEERING

Debug/Depuración: Encontrar y eliminar errores
Bug: Error
Castear: dar el formato adecuado
Parsear: Analizar un código XML sintácticamente, y revisar si tiene todos los campos que consideramos obligatorios
Anidar: Relacionar las bases de datos
On premise: trabajar de forma local (contrario a trabajar en la nube


Fromatos: Json, XML, yaml

Frameworks:
	Hadook (Spark, hive, impala)

Herramientas de ambientes:
	Virtual Machine. VirtualBox, VMWare
	Entornos virtuales
	Docker

Herramientas de accesos
	Postman
Putty
firewall


Proceso ETL:
Extracción/Ingesta: De base de datos con accesos con apis, o de bases locales
transformación con spark sobre todo o pandas (usando un procesador/cluster de una nube y un ambiente como databriks y un orquestador como data factory)
Carga sobre una base de datos propia. Almacenar en una nube como un data lake


Infraestructuras/ DW en la nube: Azure, Oracle, AMazon y Google

Azure
Databricks. Notebooks
Data lake. Repositorio
Data factory (Airflow). Orquestador 



PARA TENER ACCESO A REDES DE EMPRESAS (DONDE TIENEN CIERTOS PROGRAMAS, APLICACIONES Y DATOS), SE NECESITA contar con UN IP publica QUE ESTE DENTRO DEL RANGO DE LAS IPs DE LA RED. Varios caminos:

VPN. que biwares tenga una VPN tal que cuando nos conectamos, la VPN transforma la IP publica nuestra, en una IP valida para esa red en u proceso de encriptado o enmascarado (hay varios softwares que permiten esto. entre los más usados, Forticlient o OpenVPN). Proxy hace algo parecido pero no a nivel SO como la VPN, sino a nivel aplicación y sin software.
Tener acceso con credenciales a una maquina virtual, la cual si tenga una IP dentro del rango aceptado por la red.
Hacer una excepeción configurando el firewall, donde se solicita que las IPs nuestras, la de mi compu, tenga acceso a su red. (es lo más peliloso)

Además de todo esto, hay algunas redes privadas que tienen otra capa de seguridad y necesitan un tunnel (SSH). kerberos tambien es algo de esto y los proxy.


IP publica es la del modem a la que estoy conectado, y la que se actualiza cuando se resetea. IP privada es la que me diferencia de los demas diapositivos dentro de mi red (modem)


Luego, si me quiero conectar a la base de datos, necesito un gestor de base de datos como dbeaver y saber:
host: la url
puerto: 421
database: nombre
tipo de base de datos: mySQL o MariaDB, SQL Server
usuario y contraseña (Las credenciales)
tipo de base de datos: (si es mysql, azure, redshift,etc. cada uno de estos tiene diferentes drivers, que son quienes funcionan de interpretador, propios de cada tecnología)

