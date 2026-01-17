import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# 1. Create the folder if it doesn't exist
if not os.path.exists('model_files'):
    os.makedirs('model_files')

# 2. Define and Fit Encoders with the exact categories from your lab [cite: 57-60]
pclass_le = LabelEncoder().fit(["First", "Second", "Third"])
gender_le = LabelEncoder().fit(["Male", "Female"])
sibling_le = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
embarked_le = LabelEncoder().fit(["Southampton", "Cherbourg", "Queenstown"])

# 3. Create a Dummy Model (for testing purposes)
# In a real lab, you would train this on the Titanic dataset
svc_model = SVC()
# Simulating a fit so it can be saved
svc_model.fit([[1, 0, 0, 0], [3, 1, 1, 2]], [1, 0])

# 4. Save everything [cite: 38-50]
with open('model_files/pclass_encoder.pkl', 'wb') as f:
    pickle.dump(pclass_le, f)
with open('model_files/gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_le, f)
with open('model_files/sibling_encoder.pkl', 'wb') as f:
    pickle.dump(sibling_le, f)
with open('model_files/embarked_encoder.pkl', 'wb') as f:
    pickle.dump(embarked_le, f)
with open('model_files/svc_trained_model.pkl', 'wb') as f:
    pickle.dump(svc_model, f)

print("All 5 artifacts have been saved successfully in model_files/!")