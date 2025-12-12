from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        
        sl = float(request.form["sepal_length"])
        sw = float(request.form["sepal_width"])
        pl = float(request.form["petal_length"])
        pw = float(request.form["petal_width"])

        features = np.array([[sl, sw, pl, pw]])
        scaled_features = scaler.transform(features)
       
        pred = model.predict(scaled_features)[0]

        iris = ["Setosa", "Versicolor", "Virginica"]
        prediction = iris[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
