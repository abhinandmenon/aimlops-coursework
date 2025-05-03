import sys
from pathlib import Path
root = str(Path(__file__).parent.parent)
sys.path.append(root)

import gradio

from predict import predict_death_event, trained_model
from components import (
    in_age, in_anaemia, in_creatinine_phosphokinase, in_diabetes, in_ejection_fraction,
    in_high_blood_pressure, in_platelets, in_serum_creatinine, in_serum_sodium, in_sex,
    in_smoking, in_time, out_response
)

from fastapi import FastAPI, Response

import prometheus_client as prom

import pandas as pd
from sklearn.metrics import f1_score


title = 'Patient Survival Prediction'
description = 'Predict survival of patient with heart failure, given their clinical record'


app = FastAPI(
    title=title, openapi_url="/0.0.1/openapi.json"
)


# Gradio interface to generate UI link
iface = gradio.Interface(
    fn = predict_death_event,
    inputs = [
        in_age,
        in_anaemia,
        in_creatinine_phosphokinase,
        in_diabetes,
        in_ejection_fraction,
        in_high_blood_pressure,
        in_platelets,
        in_serum_creatinine,
        in_serum_sodium,
        in_sex,
        in_smoking,
        in_time
    ],
    outputs = [out_response],
    title = title,
    description = description,
    allow_flagging='never'
)

# iface.launch(share=False, server_name='0.0.0.0', server_port=8004)
app = gradio.mount_gradio_app(app, iface, path='/gradio')


# Metric object of type gauge
f1_metric = prom.Gauge('hfs_f1_score', 'F1 score for random 75 test samples')

# LOAD TEST DATA
test_data = pd.read_csv(root + "/test_hfs.csv")

# Function for updating metrics
def update_metrics():
    test = test_data.sample(75)
    test = test.drop('Unnamed: 0', axis=1)
    test_feat = test.drop('DEATH_EVENT', axis=1)
    test_cnt = test['DEATH_EVENT'].values
    test_pred = trained_model.predict(test_feat)
    f1 = f1_score(test_cnt, test_pred)
    
    f1_metric.set(f1)

@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 