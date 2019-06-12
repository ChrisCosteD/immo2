import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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