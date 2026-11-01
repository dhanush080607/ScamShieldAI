import joblib

# Load saved fake job model
model = joblib.load("models/fake_job_model.pkl")

print("===== ScamShield AI: Fake Job Detector =====")

while True:

    job_title = input("\nEnter job title (or exit): ")

    if job_title.lower() == "exit":
        print("Thank you!")
        break

    company = input("Enter company profile: ")
    description = input("Enter job description: ")
    requirements = input("Enter requirements: ")
    benefits = input("Enter benefits: ")

    # Combine job information
    job_text = (
        job_title + " " +
        company + " " +
        description + " " +
        requirements + " " +
        benefits
    )

    # Predict
    prediction = model.predict([job_text])[0]

    probabilities = model.predict_proba([job_text])[0]

    if prediction == 1:
        print("\nWARNING: This job posting looks FRAUDULENT!")
        confidence = probabilities[1] * 100

    else:
        print("\nThis job posting looks LEGITIMATE.")
        confidence = probabilities[0] * 100

    print(f"Model confidence: {confidence:.2f}%")

    print("\nNote: This is an AI prediction, not a guarantee.")