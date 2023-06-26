import bentoml

from bentoml.io import JSON

model_ref = bentoml.xgboost.get("credit_risk_model:xllx32yt6ovyspex")

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    prediction = model_runner.predict.run(application_data)
    return {"status": "Approved"}