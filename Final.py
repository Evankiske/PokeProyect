# Importamos las librerias a usar

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Leemos nuestros DataFrames a utilizar#

df = pd.read_csv('Pokemon.csv')
combats = pd.read_csv('Combat1.csv')
test = pd.read_csv('tests.csv')

# Creamos una copia de test
test3 = test.copy()

# Reasignamos el nombre a la columna # para facil manipulacion
df.rename(columns={'#': 'Num'}, inplace=True)

# Re Asignamos el index para el correcto manejo de la base
df.set_index(df.Num, inplace=True)

# Creamos un nuevo DataSet con las columnas que vamos a usar
poke = df.filter(["Name", "HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"], axis=1)
poke.head()

# Remplazamos los numeros con el nombre del pokemon correspondiente
cols = ["First_pokemon", "Second_pokemon", "Winner"]
fights = combats[cols].replace(df.Name)
fights.head()

# Transformamos en boleano nuestro resultado del combate
combats.Winner[combats.Winner == combats.First_pokemon] = 0
combats.Winner[combats.Winner == combats.Second_pokemon] = 1


# Creamos una funcion de normalizacion para los datos que usaremos
# Fue robada de internet y modificada por su servidor
def norm(data_df):
    stats = ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]
    stats_df = poke[stats].T.to_dict("list")
    first = data_df.First_pokemon.map(stats_df)
    second = data_df.Second_pokemon.map(stats_df)
    lista = []
    for i in range(len(first)):
        lista.append(np.array(first[i]) - np.array(second[i]))
    new_test = pd.DataFrame(lista, columns=stats)
    for c in stats:
        description = new_test[c].describe()
        new_test[c] = (new_test[c] - description['min']) / (description['max'] - description['min'])
    return new_test


# Aplicamos la funcion y creamos un nuevo DataSet
# Concatenamos las dos tablas
data = norm(combats)
data = pd.concat([data, combats.Winner], axis=1)

# Usamos la columna Winner para hacer nuestro testeo
X = data.drop("Winner", axis=1)
y = data["Winner"]

# Definimos el train y asignamos el size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Aplicamos modelo de ML para predecir (en el testeo original se usaron 5 y este fue el que mas me agrado)
clf_rfc = RandomForestClassifier(n_estimators=50)
clf_rfc.fit(X_train, y_train)
accuracies = cross_val_score(estimator=clf_rfc, X=X_train, y=y_train, cv=5, verbose=1)
y_pred = clf_rfc.predict(X_test)

#print('')
#print('####### RandomForestClassifier #######')
#print('Score : %.4f' % clf_rfc.score(X_test, y_test))
#print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

#print('')
#print('MSE    : %0.2f ' % mse)
#print('MAE    : %0.2f ' % mae)
#print('RMSE   : %0.2f ' % rmse)
#print('R2     : %0.2f ' % r2)

print('Pokemon Battle Simulator:')
print('Choose a Pokemon Between 1-809:')
First = int(input())
print('Choose a Second Pokemon Between 1-809:')
Second = int(input())

simple_list = [[First, Second]]
test2 = pd.DataFrame(simple_list, columns=['First_pokemon', 'Second_pokemon'])
test4 = test3.append(test2, ignore_index=True)
battle2 = norm(test4)
pred = clf_rfc.predict(battle2)
test4["Winner"] = [test4["First_pokemon"][i] if pred[i] == 0 else test4["Second_pokemon"][i] for i in range(len(pred))]
final1 = test4[cols].replace(poke.Name)
print(final1.iloc[10000]['First_pokemon'] + ' vs ' + final1.iloc[10000]['Second_pokemon'])
print('Winner: ' + final1.iloc[10000]['Winner'])
