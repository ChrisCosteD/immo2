import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('df.csv')

#Select only usefull data
df = df[['date_mutation','valeur_fonciere', 'surface_reelle_bati', 'longitude','latitude','type_local','nombre_pieces_principales']]

#Delete rows with NaN
df = df.dropna()

#Filter only appartments and houses
df = df.loc[(df['type_local'] == 'Appartement') | (df['type_local'] == 'Maison')]

#Calculate prix_m2
df['prix_m2'] = df.valeur_fonciere / df.surface_reelle_bati

#Exclude extreme values
df = df.loc[(df['prix_m2'] >= 1000) & (df['prix_m2'] <= 10000),:]
df = df.loc[(df['nombre_pieces_principales'] >= 1) & (df['nombre_pieces_principales'] <= 15),:]
print(len(df))


#Duplicated data
dupli_data = df[df.duplicated(['date_mutation', 'valeur_fonciere', 'longitude', 'latitude'])]

#Delete duplicated
on = ['date_mutation', 'valeur_fonciere', 'longitude', 'latitude']
nodoublon_data = (df.merge(dupli_data[on], on=on, how='left', indicator=True)
                  .query('_merge == "left_only"').drop('_merge', 1))
print(len(nodoublon_data))
#create nodoublon csv
#nodoublon_data.to_csv('nodoublon_data.csv', sep=',')

#display all columns
pd.set_option('display.max_columns', None)
#print(dupli_data)


#check duplicated on new dataframe
test = nodoublon_data[nodoublon_data.duplicated(['date_mutation', 'valeur_fonciere', 'longitude', 'latitude'])]
print(len(test))

#SET X and Y
data_X = nodoublon_data.iloc[:,2 :-5].values
data_Y = nodoublon_data.iloc[:,1 :-6].values

#Prix au m2
#data_Y = df.iloc[:,7 :].values
# Split the data into training/testing sets
data_X_train, data_X_test, data_Y_train, data_Y_test = train_test_split(data_X,data_Y, test_size=1/5, random_state=0)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, data_Y_train)

# Make predictions using the testing set
data_y_pred = regr.predict(data_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(data_Y_test, data_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(data_Y_test, data_y_pred))

# Plot outputs
plt.scatter(data_X_test, data_Y_test,  color='black')
plt.plot(data_X_test, data_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.savefig('graph.png')
