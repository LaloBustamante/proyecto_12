'''
# Descripción del proyecto  


La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y te pide que evalúes esa posibilidad.

Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.  
Tarea 2: predecir si es probable que un nuevo cliente reciba un beneficio de seguro. ¿Puede un modelo de predicción entrenado funcionar mejor que un modelo dummy no entrenado? ¿Puede funcionar peor? Explica tu respuesta.  
Tarea 3: predecir la cantidad de beneficios de seguro que probablemente recibirá un nuevo cliente utilizando un modelo de regresión lineal.  
Tarea 4: proteger los datos personales de los clientes sin romper el modelo de la tarea anterior.  

Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento de datos u ofuscación de datos. Pero los datos deben protegerse de tal manera que la calidad de los modelos de machine learning no se vea afectada. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.

### Instrucciones del proyecto  

1. arga los datos.  
2. Verifica que los datos no tengan problemas: no faltan datos, no hay valores extremos, etc.  
3. Trabaja en cada tarea y responde las preguntas planteadas en la plantilla del proyecto.  
4. Saca conclusiones basadas en tu experiencia trabajando en el proyecto.  

Hay algo de código previo en la plantilla del proyecto, siéntete libre de usarlo. Primero se debe terminar algo de código previo. Además, hay dos apéndices en la plantilla del proyecto con información útil.  

### Descripción de datos  

El dataset se almacena en el archivo /datasets/insurance_us.csv.  

Características: sexo, edad, salario y número de familiares de la persona asegurada.  
Objetivo: número de beneficios de seguro recibidos por una persona asegurada en los últimos cinco años.  

### Evaluación del proyecto  

Hemos definido los criterios de evaluación para el proyecto. Léelos con atención antes de pasar al ejercicio.

Esto es en lo que se fijarán los revisores al examinar tu proyecto:  

¿Seguiste todos los pasos de las instrucciones?  
¿Mantuviste la estructura del proyecto?  
¿Mantuviste el código ordenado?  
¿Desarrollaste todos los procedimientos necesarios y respondiste todas las preguntas?  
¿Sacaste tus conclusiones?  
'''

# Inicialización

import math
import numpy as np
import pandas as pd

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler


# Carga de datos
df = pd.read_csv('/datasets/insurance_us.csv')


# Renambrado de columnas
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


df.sample(10)
df.info()


# puede que queramos cambiar el tipo de edad (de float a int) aunque esto no es crucial

# escribe tu conversión aquí si lo deseas:

# Conversión de la columna age (edad) de float a int

df['age'] = df['age'].astype(int)


# comprueba que la conversión se haya realizado con éxito
df.info()


# ahora echa un vistazo a las estadísticas descriptivas de los datos.# ¿Se ve todo bien?

# Mostramos estadísticas descriptivas de las columnas
df.describe()


'''
# ¿Se ve todo bien?

Tras observar las estadísticas descriptivas, sí los datos parecen dentro de rangos normales para los seguros, no se observan valores 
extremos, faltantes o inconsistencias en los datos. Por lo que es posible proceder al análisis.
'''


'''
# 3  Análisis exploratorio de datos
Vamos a comprobar rápidamente si existen determinados grupos de clientes observando el gráfico de pares.
'''


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


'''
De acuerdo, es un poco complicado detectar grupos obvios (clústeres) ya que es difícil combinar diversas variables simultáneamente (para 
analizar distribuciones multivariadas). Ahí es donde LA y ML pueden ser bastante útiles.

# Tarea 1. Clientes similares

En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos. Es posible que quieras revisar las siguientes lecciones (capítulo -> lección)- Distancia entre vectores -> Distancia euclidiana

* Distancia entre vectores -> Distancia Manhattan  

Para resolver la tarea, podemos probar diferentes métricas de distancia.
'''


'''
Escribe una función que devuelva los k vecinos más cercanos para un  𝑛𝑡ℎ  
  objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de 
  prestaciones de   seguro recibidas. Puedes utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta el 
  enlace) o tu propia implementación.   Pruébalo para cuatro combinaciones de dos casos- Escalado  

* los datos no están escalados  
* los datos se escalan con el escalador MaxAbsScaler  
* Métricas de distancia  
* Euclidiana  
* Manhattan  

Responde a estas preguntas:- ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?- 
¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)? 
'''


feature_names = ['gender', 'age', 'income', 'family_members']


def get_knn(df, n, k, metric):
    
    """
    Devuelve los k vecinos más cercanos

    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar
    :param n: número de objetos para los que se buscan los vecinos más cercanos    
    :param k: número de vecinos más cercanos a devolver
    :param métric: nombre de la métrica de distancia   
   
     """
# Inicializar el modelo de vecinos más cercanos con la métrica de distancia

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)

    # Ajustar el modelo a los datos
    nbrs.fit(df[feature_names])

# Obtener las distancias y los índices de los k vecinos más cercanos
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    # Crear un DataFrame con los resultados
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# Escalar los datos

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

# Crear un nuevo DataFrame con los datos escalados
df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# Muestra aleatoria de los datos escalados
df_scaled.sample(5)


#Sin escalar y métrica Euclidiana:
print("Sin escalar, métrica Euclidiana")
result_euclidean_no_scale = get_knn(df, n=0, k=5, metric='euclidean')
display(result_euclidean_no_scale)


# Con escalado y métrica Euclidiana:
print("Con escalado, métrica Euclidiana")
result_euclidean_scaled = get_knn(df_scaled, n=0, k=5, metric='euclidean')
display(result_euclidean_scaled)


# Sin escalar y métrica Manhattan:
print("Sin escalar, métrica Manhattan")
result_manhattan_no_scale = get_knn(df, n=0, k=5, metric='manhattan')
display(result_manhattan_no_scale)


# Con escalado y métrica Manhattan:
print("Con escxalado, métrica Manhattan")
result_manhattan_scaled = get_knn(df_scaled, n=0, k=5, metric='manhattan')
display(result_manhattan_scaled)


'''
Respuestas a las preguntas

¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?

Sí, afecta significativamente. Sin escalado, las características con valores más grandes (como los ingresos) dominan las distancias, lo 
que puede sesgar los resultados. El escalado asegura que todas las características tengan el mismo peso en el cálculo de la distancia.

¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

La métrica Manhattan suele dar resultados ligeramente diferentes a la Euclidiana, ya que mide las distancias de forma distinta (suma de 
las diferencias absolutas en lugar de las diferencias cuadradas). Sin embargo, ambas métricas deberían ser razonablemente similares, 
especialmente cuando los datos están escalados.
'''


'''
# Tarea 2. ¿Es probable que el cliente reciba una prestación del seguro?

En términos de machine learning podemos considerarlo como una tarea de clasificación binaria.

Con el valor de insurance_benefits superior a cero como objetivo, evalúa si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy. Instrucciones:  

* Construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Sería interesante observar cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. Puedes utilizar una implementación ya existente del algoritmo de clasificación kNN de scikit-learn (consulta el enlace) o tu propia implementación.- Construye un modelo dummy que, en este caso, es simplemente un modelo aleatorio. Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1. La probabilidad de pagar cualquier prestación del seguro puede definirse como  

𝑃{prestación de seguro recibida}=número de clientes que han recibido alguna prestación de seguro / número total de clientes.  
 
Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.  
'''


# # сalcula el objetivo (insurance_benefits_received > 0)
df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int)


# comprueba el desequilibrio de clases con value_counts()

print(df['insurance_benefits_received'].value_counts())


'''
### Desequilibrio de Clases:  

Clase 0 (no recibió prestación): 4436 clientes (88.7%)  
Clase 1 (recibió prestación): 564 clientes (11.3%)  

Conclusión: Hay un notable desequilibrio de clases, lo que significa que muchos más clientes no reciben prestaciones en comparación con 
aquellos que sí lo hacen.
'''


# Divide los datos en entrenamiento y prueba (70:30)

X = df[['gender', 'age', 'income', 'family_members']]
y = df['insurance_benefits_received']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Escalar los datos con MaxAbsScaler

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Función para evaluar el clasificador con la métrica F1
def eval_classifier(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    print(f'F1 Score: {f1:.2f}')
    cm = confusion_matrix(y_true, y_pred, normalize='all')
    print('Matriz de confusión:')
    print(cm)


# Construir el modelo KNN y evaluar para k=1...10
for k in range(1, 11):
    print(f'--- k = {k} ---')
    
    # Modelo KNN con datos originales
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('Datos sin escalar:')
    eval_classifier(y_test, y_pred)
    
    # Modelo KNN con datos escalados
    knn.fit(X_train_scaled, y_train)
    y_pred_scaled = knn.predict(X_test_scaled)
    print('Datos escalados:')
    eval_classifier(y_test, y_pred_scaled)


'''
### Rendimiento del Clasificador kNN:  

### Datos sin escalar:  
Los resultados muestran que el F1 Score disminuye drásticamente a medida que aumenta el valor de k:  
Para k=1: F1 Score de 0.65  
Para k=2: F1 Score de 0.38  
Para k=10: F1 Score de 0.04  

Conclusión: Sin escalar los datos, el modelo de kNN tiende a sobreajustarse para valores bajos de k (como k=1), pero su desempeño se 
deteriora rápidamente a medida que k aumenta, debido a que el algoritmo no es capaz de capturar bien las diferencias en las 
características.  

### Datos escalados:  
Los datos escalados mejoran significativamente el rendimiento del modelo kNN en todos los valores de k:  
Para k=1: F1 Score de 0.93  
Para k=10: F1 Score de 0.92  

Conclusión: El escalado de los datos tiene un impacto positivo considerable en el rendimiento del modelo kNN. El rendimiento es mucho más 
estable y el F1 Score permanece alto para todos los valores de k. Esto indica que el escalado de las características es crucial para 
algoritmos basados en la distancia como kNN.
'''


# Modelo dummy: generar la salida de un modelo aleatorio

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# Probamos el modelo dummy con diferentes probabilidades
P_values = [0, df['insurance_benefits_received'].mean(), 0.5, 1]
for P in P_values:
    print(f'\nProbabilidad: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, len(y_test))
    eval_classifier(y_test, y_pred_rnd)
    print()


'''
### Rendimiento del Modelo Dummy:  

Probamos el modelo dummy con diferentes probabilidades (P=0, P=0.11, P=0.50, P=1), y obtuvimos los siguientes F1 Scores:

P=0: F1 Score de 0.00 (ninguna prestación predicha)  
P=0.11 (la proporción real de prestaciones): F1 Score de 0.17  
P=0.50: F1 Score de 0.20  
P=1: F1 Score de 0.19 (siempre predice que se recibirá una prestación)  

Conclusión: El modelo dummy muestra que el rendimiento es bajo para todas las probabilidades probadas. El mejor resultado (F1 Score de 
0.20) se alcanza cuando se utiliza una probabilidad de 0.50, pero sigue siendo considerablemente inferior al modelo kNN con datos 
escalados.
'''


'''
En general, observamos que para los datos sin escalar, los modelos tienden a predecir un número mucho mayor de casos negativos (clientes 
que no reciben prestaciones), lo que refleja el desequilibrio de clases.  

En los datos escalados, el modelo kNN realiza predicciones mucho más equilibradas, con tasas de error más bajas en ambas clases, lo que 
lleva a un mejor F1 Score.  

Escalado es crucial: El escalado de características es absolutamente necesario para que el modelo kNN funcione correctamente, ya que 
mejora drásticamente el rendimiento.  

k=1-5 son buenos valores: En general, los mejores valores de k están entre 1 y 5 para los datos escalados, ya que proporcionan un 
equilibrio entre precisión y generalización.  

Modelo dummy es ineficiente: El modelo dummy, incluso en su mejor configuración (P=0.5), tiene un desempeño muy inferior al modelo kNN. Esto demuestra que kNN es una mejor opción para este problema de clasificación.
'''

'''
# Tarea 3. Regresión (con regresión lineal)

Con insurance_benefits como objetivo, evalúa cuál sería la RECM de un modelo de regresión lineal.

Construye tu propia implementación de regresión lineal. Para ello, recuerda cómo está formulada la solución de la tarea de regresión lineal en términos de LA. Comprueba la RECM tanto para los datos originales como para los escalados. ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?  

Denotemos-  𝑋: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades-  𝑦 — objetivo (un vector)-  𝑦̂ — objetivo estimado (un vector)-  𝑤 — vector de pesos La tarea de regresión lineal en el lenguaje de las matrices puede formularse así:  

𝑦=𝑋𝑤  

El objetivo de entrenamiento es entonces encontrar esa  𝑤  w que minimice la distancia L2 (ECM) entre  𝑋𝑤  y 𝑦:  

min𝑤𝑑2(𝑋𝑤,𝑦) or min𝑤MSE(𝑋𝑤,𝑦)  
 

Parece que hay una solución analítica para lo anteriormente expuesto:  

𝑤=(𝑋𝑇𝑋)−1𝑋𝑇𝑦  
 

La fórmula anterior puede servir para encontrar los pesos  𝑤 y estos últimos pueden utilizarse para calcular los valores predichos  

𝑦̂ =𝑋𝑣𝑎𝑙𝑤  
'''


# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        # Calcular los pesos usando la fórmula de regresión lineal (solución analítica)
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y

    def predict(self, X):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        # Calcular los valores predichos (ŷ = Xw)
        y_pred = X2 @ self.weights
        
        return y_pred
    

# Función para evaluar el modelo de regresión
def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    


# Datos de características y objetivo
X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

# División de datos en entrenamiento y prueba (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Inicializar el modelo de regresión
lr = MyLinearRegression()

# Entrenar el modelo con los datos de entrenamiento
lr.fit(X_train, y_train)

# Imprimir los pesos aprendidos
print("Pesos del modelo:", lr.weights)

# Predecir en los datos de prueba
y_test_pred = lr.predict(X_test)

# Evaluar el rendimiento del modelo en los datos de prueba
eval_regressor(y_test, y_test_pred)


'''
### Comparar el RMSE de los datos originales y escalados:  

Aquí se prueba la regresión con los datos escalados (normalizando o estandarizando las características), para ver si hay alguna diferencia en el RMSE.
'''


# Escalado de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos escalados
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=12345)

# Entrenar el modelo en los datos escalados
lr.fit(X_train_scaled, y_train)

# Predecir en los datos de prueba escalados
y_test_pred_scaled = lr.predict(X_test_scaled)

# Evaluar el modelo con los datos escalados
eval_regressor(y_test, y_test_pred_scaled)


'''
### RMSE (Raíz del Error Cuadrático Medio):  

El RMSE de 0.34 indica que, en promedio, la diferencia entre los valores predichos por el modelo y los valores reales es aproximadamente 0.34 unidades. Esto implica que el modelo tiene un rendimiento razonablemente bueno para predecir la cantidad de beneficios de seguro.  

Si observamos los valores de RMSE para los datos originales y los escalados, podemos ver que ambos resultaron en un RMSE de 0.34. Esto sugiere que, en este caso específico, el escalado no tuvo un impacto significativo en el rendimiento del modelo de regresión. Esto puede deberse a que las características no varían drásticamente en magnitud y por lo tanto, el modelo es capaz de ajustarse bien incluso sin escalarlas.  

'''


'''
# Tarea 4. Ofuscar datos

Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz  𝑋) por una matriz invertible  𝑃.  

𝑋′=𝑋×𝑃  
 

Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto, la propiedad de invertibilidad es importante aquí, así que asegúrate de que  𝑃 sea realmente invertible.  

Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla de multiplicación de matrices y su implementación con NumPy.  

'''


# Extraer las características numéricas de la columna de información personal
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# Convertir a matriz NumPy
X = df_pn.to_numpy()


# Generar matriz aleatoria P
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))


# Verificar si P es invertible calculando su determinante
if np.linalg.det(P) != 0:
    print("La matriz P es invertible.")
else:
    print("La matriz P no es invertible. Generando otra matriz.")

# Invertir la matriz para asegurarnos que podemos recuperar los datos más tarde
P_inv = np.linalg.inv(P)

# Transformar los datos originales
X_transformed = np.dot(X, P)

# Recuperar los datos originales
X_recovered = np.dot(X_transformed, P_inv)

# Mostrar los tres casos para algunos clientes
for i in range(3):
    print(f"Cliente {i+1}:")
    print(f"Original: {X[i]}")
    print(f"Transformado: {X_transformed[i]}")
    print(f"Recuperado: {X_recovered[i]}\n")

# Comparar los datos recuperados con los originales
difference = np.abs(X - X_recovered)
print(f"Diferencia promedio entre datos originales y recuperados: {np.mean(difference)}")


'''
¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?

No, después de aplicar la transformación con la matriz 𝑃, los valores de las características (como la edad y los ingresos) se ven significativamente alterados y no tienen ninguna correlación directa con los valores originales, lo que los hace ininteligibles. Esto garantiza la ofuscación de los datos.

¿Puedes recuperar los datos originales de  𝑋′ si conoces  𝑃 ? Intenta comprobarlo a través de los cálculos moviendo  𝑃 del lado derecho de la fórmula anterior al izquierdo. En este caso las reglas de la multiplicación matricial son realmente útiles.

Sí, es posible recuperar los datos originales si conocemos la matriz P y esta es invertible. Esto se puede lograr multiplicando los datos ofuscados por la inversa de P. De esta manera, aplicamos la operación inversa de la transformación.Sí, puedes recuperar los datos originales multiplicando los datos transformados $𝑋′$ por la inversa de $P(P^-1)$. Esto se debe a que $𝑋′ = 𝑋 × 𝑃$, y cuando multiplicamos por $P^−1$, se cancela la transformación y obtenemos los datos originales.

Muestra los tres casos para algunos clientes- Datos originales  
  
El que está transformado- El que está invertido (recuperado)  

El código muestra los valores originales, los transformados, y los recuperados para tres clientes de ejemplo. Puedes ver cómo los valores se ofuscan en la transformación y se recuperan correctamente al aplicar la matriz inversa.

Seguramente puedes ver que algunos valores no son exactamente iguales a los de los datos originales. ¿Cuál podría ser la razón de ello?

Las pequeñas diferencias que pueden aparecer entre los datos originales y los recuperados generalmente se deben a errores de precisión numérica inherentes a los cálculos con matrices en punto flotante. Aunque estas diferencias suelen ser mínimas, es algo común cuando se trabaja con matrices invertibles y operaciones de álgebra lineal en computadoras.

'''


'''
# 4  Prueba de que la ofuscación de datos puede funcionar con regresión lineal

En este proyecto la tarea de regresión se ha resuelto con la regresión lineal. Tu siguiente tarea es demostrar analytically que el método de ofuscación no afectará a la regresión lineal en términos de valores predichos, es decir, que sus valores seguirán siendo los mismos. ¿Lo puedes creer? Pues no hace falta que lo creas, ¡tienes que que demostrarlo!

Entonces, los datos están ofuscados y ahora tenemos  𝑋×𝑃 en lugar de tener solo  𝑋. En consecuencia, hay otros pesos  𝑤𝑃 como  

𝑤=(𝑋𝑇𝑋)−1𝑋𝑇𝑦⇒𝑤𝑃=[(𝑋𝑃)𝑇𝑋𝑃]−1(𝑋𝑃)𝑇𝑦  
 

¿Cómo se relacionarían  𝑤 y 𝑤𝑃 si simplificáramos la fórmula de  𝑤𝑃 anterior?  

¿Cuáles serían los valores predichos con  𝑤𝑃?  

¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM? Revisa el Apéndice B Propiedades de las matrices al final del cuaderno. ¡Allí encontrarás fórmulas muy útiles!  

No es necesario escribir código en esta sección, basta con una explicación analítica.  

### ¿Cómo se relacionarían  𝑤 y 𝑤𝑃 si simplificáramos la fórmula de  𝑤𝑃 anterior?  

Observamos que la expresión (𝑋<sup>𝑇</sup>𝑋)<sup>−1</sup>𝑋<sup>𝑇</sup>𝑦 es precisamente el vector de pesos original $𝑤$. Entonces, podemos expresar el vector de pesos $𝑤𝑃$ de la siguiente manera:  

𝑤𝑃=𝑃<sup>-1</sup>𝑤  

### ¿Cuáles serían los valores predichos con  𝑤𝑃?  

Los valores predichos con el nuevo modelo ofuscado serían:

$𝑦′=𝑋′𝑤𝑃=(𝑋×𝑃)𝑤𝑃y$  

Sustituyendo  𝑤𝑃=𝑃<sup>−1</sup>:  

𝑦′=(𝑋×𝑃)(𝑃<sup>−1</sup>𝑤)  

Dado que 𝑃×𝑃<sup>−1</sup>=𝐼, tenemos:  

$𝑦′=𝑋𝑤$

Conclusión:  

Los valores predichos 𝑦′ con los datos ofuscados 𝑋×𝑃 son idénticos a los valores predichos 𝑦 con los datos originales 𝑋. Esto demuestra que la ofuscación no afecta en absoluto a los valores predichos por el modelo.

### ¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM?  

Dado que los valores predichos no cambian tras la ofuscación de los datos, la métrica de calidad como la RECM (Raíz del Error Cuadrático Medio) también permanecerá igual. La RECM se calcula comparando los valores reales 𝑦 y con los valores predichos 𝑦', y dado que 𝑦′= y, el error seguirá siendo el mismo, independientemente de la ofuscación.  

En conclusión, la ofuscación mediante multiplicación por una matriz invertible no tiene ningún efecto en los valores predichos ni en la calidad de la regresión medida por la RECM u otras métricas de error.

'''


'''
Prueba analítica

### Demostración analítica de que la ofuscación de datos no afecta a la regresión lineal:  

En este ejercicio, tenemos una regresión lineal donde los datos originales X se transforman mediante una matriz invertible P, y el objetivo es demostrar que esta ofuscación no afecta los valores predichos en la regresión lineal. 

### Regresión lineal sin ofuscación:  

Sabemos que el vector de pesos 𝑤 en una regresión lineal se calcula utilizando la siguiente fórmula:  

𝑤=(𝑋<sup>𝑇</sup>𝑋)<sup>^−1</sup>𝑋<sup>𝑇</sup>𝑦  

Donde:  

* X es la matriz de características (original),  
* y es el vector objetivo (valores reales que intentamos predecir).  


Regresión lineal con ofuscación:  

Si ofuscamos los datos, multiplicamos X por una matriz invertible P, obteniendo una nueva matriz de características $𝑋′=𝑋×𝑃$. Ahora, el vector de pesos cambia a wp y se calcula de la siguiente forma:  

𝑤𝑃=((𝑋×𝑃)<sup>𝑇</sup>(𝑋×𝑃))<sup>−1</sup>(X×P)<sup>𝑇</sup>y$  

Simplifiquemos esta expresión paso a paso:  

Transpuesta del producto:  

(𝑋×𝑃)<sup>𝑇</sup>=𝑃<sup>𝑇</sup>𝑋<sup>𝑇</sup>  
 
Sustituyendo en la ecuación de  𝑤𝑃:  

𝑤𝑃=(𝑃<sup>𝑇</sup>𝑋<sup>𝑇</sup>𝑋𝑃)<sup>−1</sup>(𝑃<sup>𝑇</sup>𝑋<sup>𝑇</sup>𝑦)  

Propiedad de la inversa de un producto de matrices:  

Sabemos que (𝐴𝐵𝐶)<sup>−1</sup>=𝐶<sup>−1</sup>𝐵<sup>−1</sup>𝐴<sup>−1</sup>, por lo que:  

𝑤𝑃=𝑃<sup>−1</sup>(𝑋<sup>𝑇</sup>𝑋)<sup>−1</sup>(𝑃<sup>𝑇</sup>)<sup>−1</sup>P<sup>T</sup>X<sup>T</sup>y  

Dado que: (𝑃<sup>𝑇</sup>)<sup>−1</sup>𝑃<sup>𝑇</sup>=𝐼 (la matriz identidad), podemos simplificar aún más:  

𝑤𝑃=𝑃<sup>−1</sup>(𝑋<sup>𝑇</sup>𝑋)<sup>−1</sup>𝑋<sup>𝑇</sup>𝑦$

'''


'''
# 5  Prueba de regresión lineal con ofuscación de datos

Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida. Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una implementación de regresión lineal de scikit-learn o tu propia implementación. Ejecuta la regresión lineal para los datos originales y los ofuscados, compara los valores predichos y los valores de las métricas RMSE y  𝑅2  
 . ¿Hay alguna diferencia?  

Procedimiento  
  
* Crea una matriz cuadrada  𝑃 de números aleatorios.- Comprueba que sea invertible. Si no lo es, repite el primer paso hasta obtener una matriz   
invertible.- <¡ tu comentario aquí !>  
* Utiliza  𝑋𝑃 como la nueva matriz de características  
'''


# Extraer las características numéricas de la columna de información personal
personal_info_column_list = ['age', 'income', 'family_members']  # Excluimos 'gender' si es categórica
df_pn = df[personal_info_column_list]

# Convertir a matriz NumPy
X = df_pn.to_numpy()
y = df['insurance_benefits'].to_numpy()

# Función para evaluar el modelo
def evaluate_model(X, y, description="Modelo original"):
    # Instanciar el modelo de regresión lineal
    model = LinearRegression()
    
    # Ajustar el modelo
    model.fit(X, y)
    
    # Predecir los valores
    y_pred = model.predict(X)
    
    # Calcular las métricas RMSE y R²
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"{description}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    return y_pred, rmse, r2


# Evaluar el modelo con los datos originales
y_pred_original, rmse_original, r2_original = evaluate_model(X, y, description="Modelo con datos originales")

# Generar una matriz aleatoria P
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

# Verificar si P es invertible calculando su determinante
while np.linalg.det(P) == 0:
    # Si P no es invertible, generar otra matriz
    P = rng.random(size=(X.shape[1], X.shape[1]))

print("La matriz P es invertible.")

# Ofuscar los datos
X_obfuscated = X @ P  # Multiplicación de matrices


# Evaluar el modelo con los datos ofuscados
y_pred_obfuscated, rmse_obfuscated, r2_obfuscated = evaluate_model(X_obfuscated, y, description="Modelo con datos ofuscados")

# Comparamos los resultados
print("\nComparación de métricas entre datos originales y ofuscados:")
print(f"RMSE original: {rmse_original:.4f}, RMSE ofuscado: {rmse_obfuscated:.4f}")
print(f"R² original: {r2_original:.4f}, R² ofuscado: {r2_obfuscated:.4f}")

# Verificar que los valores predichos sean iguales (dentro de la tolerancia numérica)
if np.allclose(y_pred_original, y_pred_obfuscated):
    print("Los valores predichos son iguales para los datos originales y ofuscados.")
else:
    print("Los valores predichos son diferentes entre los datos originales y ofuscados.")


'''
# Conclusiones

Preservación del rendimiento del modelo:  
Los resultados obtenidos muestran que tanto el RMSE (error cuadrático medio) como el R² (coeficiente de determinación) son idénticos para los modelos entrenados con los datos originales y los datos ofuscados.  
Esto confirma que la ofuscación de los datos no afecta el rendimiento del modelo de regresión lineal. Las métricas de calidad del modelo se mantienen iguales, lo que indica que las predicciones son consistentes independientemente de si los datos han sido transformados (ofuscados) o no.

Equivalencia de las predicciones:  
Los valores predichos por el modelo con los datos originales y los datos ofuscados son iguales dentro de la tolerancia numérica, lo que refuerza la conclusión de que la ofuscación mediante la multiplicación por una matriz invertible no cambia la capacidad del modelo para realizar predicciones precisas.  

Este comportamiento se debe a que la ofuscación solo afecta la representación de los datos, pero no altera las relaciones subyacentes entre las variables independientes (características) y la variable dependiente (objetivo).

Aplicación efectiva de la ofuscación:  
La técnica de ofuscación usando una matriz invertible preserva la confidencialidad de los datos originales, ya que los valores de las características han sido transformados y no pueden ser interpretados fácilmente sin la matriz invertida.  
Sin embargo, esta transformación no afecta el proceso de aprendizaje del modelo ni las predicciones, lo que hace que sea un método efectivo para proteger los datos mientras se mantiene la utilidad para el análisis de regresión.

Impacto en la calidad de la regresión:  
Dado que las métricas RMSE y R² son idénticas, podemos concluir que la calidad de la regresión lineal no se ve afectada por la ofuscación de los datos. Esto es coherente con el análisis teórico realizado anteriormente, donde mostramos que la multiplicación por una matriz invertible no altera los resultados del ajuste del modelo.
'''

