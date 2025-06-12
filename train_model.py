import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Define the complete set of labels (must match the ones used to generate the dataset)
interest_labels = ['coding', 'analysis', 'leadership', 'people', 'building', 'logic',
                   'problem solving', 'planning', 'business', 'analytics', 'organizing',
                   'debugging', 'creativity', 'innovation', 'strategy', 'negotiation',
                   'system design', 'statistics', 'project management', 'marketing', 
                   'networking', 'design']

subject_labels = ['cs', 'math', 'economics', 'psychology', 'physics', 'stats', 'management',
                  'finance', 'english', 'business', 'it', 'art', 'design', 'ai', 'sociology']

# Load dataset
df = pd.read_csv("careers_data.csv")

# Convert Interests and Subjects to lists
df["Interests"] = df["Interests"].apply(lambda x: [i.strip().lower() for i in x.split(",")])
df["Subjects"] = df["Subjects"].apply(lambda x: [i.strip().lower() for i in x.split(",")])

# Encode with consistent label sets
mlb_interests = MultiLabelBinarizer(classes=interest_labels)
mlb_subjects = MultiLabelBinarizer(classes=subject_labels)

encoded_interests = mlb_interests.fit_transform(df["Interests"])
encoded_subjects = mlb_subjects.fit_transform(df["Subjects"])

# Collect skill ratings
skills = df[["Communication", "Programming", "Management", "Creativity"]].values

# Combine all features
X = pd.concat([
    pd.DataFrame(encoded_interests),
    pd.DataFrame(encoded_subjects),
    pd.DataFrame(skills)
], axis=1)

# Target label
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "career_model.pkl")

print("✅ Model trained and saved as 'career_model.pkl'")
