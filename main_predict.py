import class_dataframe as df_class
import class_training as training
from flask_api import FlaskAPI
from flask import jsonify
from flask import Flask, render_template, request

df_object = df_class.PdDataframe('df.csv')
df_object.filter()
df = df_object.clean
# print(len(df))
# print("df = ", df.head())
# train_object = training.Training(df)
# train_object.linear_regression('[:,2 :-5]', '[:,1 :-6]', 'TEST')
train_object2 = training.Training(df)



app = FlaskAPI(__name__)

@app.route('/')
def student():
    return render_template('predict_form.html')


@app.route('/api', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        prixpredict = int(train_object2.multi_regression('valeur_fonciere', ['surface_reelle_bati', 'nombre_pieces_principales'],
                                                int(request.form["surface_reelle_bati"]),
                                                int(request.form["nombre_pieces_principales"])))
        return render_template("predict_result.html", result=prixpredict)

@app.route("/predict/<surface>,<nb_piece>,<lat>,<lon>/", methods=["GET"])
def requete_prix(surface, nb_piece, lat, lon):
    dico = {}
    #prixpredict = int(train_object2.multi_regression('valeur_fonciere', ['surface_reelle_bati', 'nombre_pieces_principales'],
     #                                  int(surface),
      #                                 int(nb_piece)))
    prixpredict = train_object2.skMultiRegression(int(surface), int(nb_piece), float(lat), float(lon))
    dico["surface_reelle_bati"] = surface
    dico["nombre_pieces_principales"] = nb_piece
    dico["result"] = prixpredict
    return jsonify(dico)


app.run(debug=True)
