import class_dataframe as df_class
import class_training as training
from flask_api import FlaskAPI
from flask import jsonify
from flask import render_template, request

df_object = df_class.PdDataframe('df.csv')
df_object.filter()
df = df_object.clean
train_object2 = training.Training(df)


app = FlaskAPI(__name__)

@app.route('/')
def student():
    return render_template('predict_form.html')


@app.route('/resultats', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        prixpredict = int(train_object2.multi_regression('valeur_fonciere', ['surface_reelle_bati', 'nombre_pieces_principales'],
                                                int(request.form["superficie"]),
                                                int(request.form["nb_pieces"])))
        return render_template("predict_result.html", result=prixpredict)


@app.route("/api", methods=["GET", "POST"])

def requete_prix():

    if request.method == 'POST':
        superficie = int(request.data.get('superficie', ''))
        nb_piece = int(request.data.get('nb_pieces', ''))
        lat = float(request.data.get('latitude', ''))
        lon = float(request.data.get('longitude', ''))


    elif request.method == 'GET':
        superficie = int(request.args.get('superficie'))
        nb_piece = int(request.args.get('nb_pieces'))
        lat = float(request.args.get('latitude'))
        lon = float(request.args.get('longitude'))

    prixpredict = int(train_object2.skMultiRegression(int(superficie), int(nb_piece), float(lat), float(lon)))

    return jsonify({'reponse': f"Le prix du bien est de {prixpredict} euros"})

app.run(debug=True)