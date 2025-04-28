import joblib


# Load trained model
trained_model_file_name = 'heart_failure_model.pkl'
trained_model = joblib.load(filename=trained_model_file_name)


# Function for prediction
def predict_death_event(
        age,
        anaemia,
        creatinine_phosphokinase,
        diabetes,
        ejection_fraction,
        high_blood_pressure,
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        smoking,
        time
    ):

    input = [
        age,
        0 if anaemia == 'No' else 1,
        creatinine_phosphokinase,
        0 if diabetes == 'No' else 1,
        ejection_fraction,
        0 if high_blood_pressure == 'No' else 1,
        platelets,
        serum_creatinine,
        serum_sodium,
        0 if sex == 'Female' else 1,
        0 if smoking == 'No' else 1,
        time
    ]

    pred = trained_model.predict([input])
    return 'Not At Risk' if pred[0] == 0 else 'At Risk'
