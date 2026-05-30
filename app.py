from flask import Flask, request, render_template

from src.pipeline import predict_pipeline
from src.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)

application = Flask(__name__)

app = application


@app.route("/")
def index():

    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":

        return render_template("home.html")

    else:

        data = CustomData(

            gender=request.form.get("gender"),

            SeniorCitizen=int(
                request.form.get("SeniorCitizen")
            ),

            Partner=request.form.get("Partner"),

            Dependents=request.form.get("Dependents"),

            tenure=float(
                request.form.get("tenure")
            ),

            PhoneService=request.form.get("PhoneService"),

            MultipleLines=request.form.get("MultipleLines"),

            InternetService=request.form.get("InternetService"),

            OnlineSecurity=request.form.get("OnlineSecurity"),

            OnlineBackup=request.form.get("OnlineBackup"),

            DeviceProtection=request.form.get("DeviceProtection"),

            TechSupport=request.form.get("TechSupport"),

            StreamingTV=request.form.get("StreamingTV"),

            StreamingMovies=request.form.get("StreamingMovies"),

            Contract=request.form.get("Contract"),

            PaperlessBilling=request.form.get("PaperlessBilling"),

            PaymentMethod=request.form.get("PaymentMethod"),

            MonthlyCharges=float(
                request.form.get("MonthlyCharges")
            ),

            TotalCharges=float(
                request.form.get("TotalCharges")
            )
        )

        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()

        prediction, probability = predict_pipeline.predict(pred_df)
        churn_probability = round(probability[0][1] * 100,2)

        if prediction[0] == 1:

            prediction_text = (
                f"Customer Will Churn "
                f"({churn_probability}% probability)")

        else:

            prediction_text = (
                f"Customer Will Not Churn "
                f"({100-churn_probability}% confidence)"
            )
        

        return render_template(

            "home.html",

            results=prediction_text,

            probability=churn_probability,

            tenure=request.form.get("tenure"),

            monthly=request.form.get("MonthlyCharges"),

            total=request.form.get("TotalCharges")
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)