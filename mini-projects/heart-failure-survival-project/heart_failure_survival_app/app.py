import gradio

from predict import predict_death_event
from components import (
    in_age, in_anaemia, in_creatinine_phosphokinase, in_diabetes, in_ejection_fraction,
    in_high_blood_pressure, in_platelets, in_serum_creatinine, in_serum_sodium, in_sex,
    in_smoking, in_time, out_response
)


# Gradio interface to generate UI link
title = 'Patient Survival Prediction'
description = 'Predict survival of patient with heart failure, given their clinical record'

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

iface.launch(share=False, server_name='0.0.0.0', server_port=8004)
