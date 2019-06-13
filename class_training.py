import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math


class Training:

    def __init__(self, dataframe):
        self.df = dataframe

    def linear_regression(self, x_iloc, y_iloc, figure_name, testsize = 1 / 5, randomstate = 0):
        # SET X and Y
        data_XInstruction = "self.df.iloc"+x_iloc+".values"
        data_YInstruction = "self.df.iloc" + y_iloc + ".values"
        data_X = eval(data_XInstruction)
        data_Y = eval(data_YInstruction)

        # Split the data into training/testing sets
        data_X_train, data_X_test, data_Y_train, data_Y_test = train_test_split(data_X,
                                                                                data_Y,
                                                                                test_size=testsize,
                                                                                random_state=randomstate)
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(data_X_train, data_Y_train)
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
        plt.scatter(data_X_test, data_Y_test, color='black')
        plt.plot(data_X_test, data_y_pred, color='blue', linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.savefig(figure_name + '.png')


    def multi_regression(self, argY, argX, taille, nb_piece):
        Y = self.df[argY]
        X = self.df[argX]
        # Ordinary Least Squares (OLS)
        est = sm.OLS(Y, X).fit()
        result_coef = est.params
        coef_surface = result_coef[0]
        coef_piece = result_coef[1]
        #print("resul =", round(coef_surface*taille + coef_piece*nb_piece, 2), "euros")
        return coef_surface*taille + coef_piece*nb_piece

    @staticmethod
    def convertLatLonToDist(lat, lon, type):
        if type == 'A':
            latA = 45.762
            lonA = 4.827
        elif type == 'B':
            latA = 45.757
            lonA = 4.832
        elif type == 'C':
            latA = 45.771
            lonA = 4.853
        else:
            print("Error convertLatLonToDist")
            return "Error"
        x = (lon - lonA) * math.cos((latA + lat) / 2)
        y = lat - latA
        d = math.sqrt((x * x) + (y * y)) * 6371
        return d

    def skMultiRegression(self, surface, nbPieces, lat, lon):
        dataset = pandas.read_csv("df.csv")
        dataframe = pandas.DataFrame(dataset,
                                     columns=['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
                                              'distanceA', 'distanceB', 'distanceC'])

        y = dataframe['valeur_fonciere']

        x = dataframe[['surface_reelle_bati', 'nombre_pieces_principales', 'distanceA', 'distanceB', 'distanceC']]

        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=1/10, random_state=0)


        regr = linear_model.LinearRegression()

        regr.fit(xTrain, yTrain)
        myDistA = self.convertLatLonToDist(lat, lon, 'A')
        myDistB = self.convertLatLonToDist(lat, lon, 'B')
        myDistC = self.convertLatLonToDist(lat, lon, 'C')

        predictMe = np.array([[surface, nbPieces, myDistA, myDistB, myDistC]])

        result = regr.predict(predictMe)

        return result[0]


