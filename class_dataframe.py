import pandas as pd


class PdDataframe:

    def __init__(self, csv_file):
        self.csv_file = pd.read_csv(csv_file)
        self.df = ""
        self.duplicated = ""
        self.clean = ""

        # display all columns
        pd.set_option('display.max_columns', None)

    def filter(self):

        self.df = self.csv_file[['date_mutation', 'valeur_fonciere', 'surface_reelle_bati', 'longitude',
                                 'latitude', 'type_local', 'nombre_pieces_principales']]

        # Delete rows with NaN
        self.df = self.df.dropna()

        # Filter only appartments and houses
        self.df = self.df.loc[(self.df['type_local'] == 'Appartement') |
                              (self.df['type_local'] == 'Maison')]

        # Add prix_m2
        self.df['prix_m2'] = self.df.valeur_fonciere / self.df.surface_reelle_bati

        # Exclude extreme values
        self.df = self.df.loc[(self.df['prix_m2'] >= 2000) &
                              (self.df['prix_m2'] <= 10000), :]

        self.df = self.df.loc[(self.df['surface_reelle_bati'] >= 10) &
                              (self.df['surface_reelle_bati'] <= 350), :]

        self.df = self.df.loc[(self.df['nombre_pieces_principales'] >= 1) &
                              (self.df['nombre_pieces_principales'] <= 15), :]

        # Duplicated data
        self.duplicated = self.df[self.df.duplicated(['date_mutation', 'valeur_fonciere', 'longitude', 'latitude'])]

        # Delete duplicated
        on = ['date_mutation', 'valeur_fonciere', 'longitude', 'latitude']
        self.clean = (self.df.merge(self.duplicated[on], on=on, how='left', indicator=True)
                          .query('_merge == "left_only"').drop('_merge', 1))
