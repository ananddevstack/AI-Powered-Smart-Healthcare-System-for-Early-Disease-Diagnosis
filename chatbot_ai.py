def ai_response(user_msg, disease=None):
    user_msg = user_msg.lower()

    if "hello" in user_msg or "hi" in user_msg:
        return "Hello! I am your AI Health Assistant. How can I help you?"

    if "diabetes" in user_msg:
        return "Diabetes is a chronic condition. Maintain healthy diet, exercise regularly and monitor blood sugar."

    if "heart" in user_msg:
        return "Heart disease can be reduced by avoiding smoking, exercising, and controlling cholesterol."

    if "cancer" in user_msg:
        return "Early detection of cancer improves survival. Regular checkups are advised."

    if "precaution" in user_msg or "care" in user_msg:
        if disease == "Diabetes":
            return "Avoid sugar, exercise daily, monitor glucose levels."
        if disease == "Heart":
            return "Avoid fatty foods, exercise, control BP."
        if disease == "Cancer":
            return "Follow doctor advice and attend regular screenings."

    if "doctor" in user_msg:
        return "You can chat with a doctor using the Doctor Chat option."

    return "Please consult a doctor for detailed diagnosis. I can give general guidance only."
