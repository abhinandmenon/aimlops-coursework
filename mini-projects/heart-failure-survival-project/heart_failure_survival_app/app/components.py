import gradio


# Inputs from user
in_age = gradio.Slider(minimum=40, maximum=95, label='Age')
in_anaemia = gradio.Radio(choices=['Yes', 'No'], label='Has Anaemia')
in_creatinine_phosphokinase = gradio.Slider(minimum=23, maximum=1280.25, label='Level of CPK enzyme')
in_diabetes = gradio.Radio(choices=['Yes', 'No'], label='Has Diabetes')
in_ejection_fraction = gradio.Slider(minimum=14, maximum=67.5, label='Ejection Fraction')
in_high_blood_pressure = gradio.Radio(choices=['Yes', 'No'], label='Has High Blood Pressure')
in_platelets = gradio.Slider(minimum=81478, maximum=440000, label='Platelet Count')
in_serum_creatinine = gradio.Slider(minimum=0.44, maximum=2.15, label='Serum Creatinine')
in_serum_sodium = gradio.Slider(minimum=4, maximum=148, label='Serum Sodium')
in_sex = gradio.Radio(choices=['Male', 'Female'], label='Gender')
in_smoking = gradio.Radio(choices=['Yes', 'No'], label='Smoker')
in_time = gradio.Slider(minimum=4, maximum=285, label='Follow-up period')

# Output response
out_response = gradio.Radio(choices=['At Risk', 'Not At Risk'])
