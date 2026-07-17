import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000  # synthetic samples to add


def gen_depression(row):
    score = 0
    score += row['Do you have Anxiety?'] * 4
    score += row['Do you have Panic attack?'] * 3.5
    score += max(0, row['Stress Levels'] - 5) * 1.5
    score += max(0, 5 - row['Work-Life Balance']) * 1.5
    score += row['Family History of Mental Illness'] * 3
    score += (row['Substance Use'] in [1, 2]) * 2.5
    score += (row['Exercise Frequency'] == 0) * 2
    score += (row['Sleep Schedule'] == 0) * 2
    score += (row['Age'] > 40) * 0.5
    prob = min(score / 20, 0.95)
    # make no-depression cases clearly negative
    if score < 3:
        prob = 0.05
    return int(np.random.random() < prob)


genders = [0, 1, 2]  # Female, Male, Other
sleep = [0, 1]       # Irregular, Regular
exercise = [0, 1, 2]  # Never, Occasional, Regular
substance = [0, 1, 2]  # None, Occasional, Frequent

rows = []
for _ in range(n):
    row = {
        'Age': np.random.randint(18, 60),
        'Gender': np.random.choice(genders, p=[0.5, 0.4, 0.1]),
        'Daily Screen Time (hours)': round(np.random.uniform(1, 12), 2),
        'Do you have Anxiety?': np.random.choice([0, 1], p=[0.55, 0.45]),
        'Do you have Panic attack?': np.random.choice([0, 1], p=[0.65, 0.35]),
        'Sleep Schedule': np.random.choice(sleep),
        'Exercise Frequency': np.random.choice(exercise),
        'Family History of Mental Illness': np.random.choice([0, 1], p=[0.75, 0.25]),
        'Substance Use': np.random.choice(substance, p=[0.6, 0.3, 0.1]),
        'Stress Levels': np.random.randint(1, 11),
        'Work-Life Balance': np.random.randint(1, 11),
    }
    row['Do you have Depression?'] = gen_depression(row)
    # reverse encode for CSV compatibility
    row['Gender'] = ['Female', 'Male', 'Other'][row['Gender']]
    row['Sleep Schedule'] = ['Irregular', 'Regular'][row['Sleep Schedule']]
    row['Exercise Frequency'] = ['Never', 'Occasional',
                                 'Regular'][row['Exercise Frequency']]
    row['Substance Use'] = ['None', 'Occasional',
                            'Frequent'][row['Substance Use']]
    row['Family History of Mental Illness'] = row['Family History of Mental Illness']
    mood_prob = row['Do you have Depression?']
    probs = np.array([0.5 + mood_prob * 0.3, 0.3, 0.2 - mood_prob * 0.1])
    probs = probs / probs.sum()  # normalize to sum to 1
    row['Mood'] = np.random.choice(['Sad', 'Neutral', 'Happy'], p=probs)
    rows.append(row)

synth = pd.DataFrame(rows)

# Load original
orig = pd.read_csv('mental_health_data.csv')

# Combine
combined = pd.concat([orig, synth], ignore_index=True)
combined.to_csv('mental_health_data_augmented.csv', index=False)
print(f"Done! Total rows: {len(combined)}")
print(
    f"Depression distribution:\n{combined['Do you have Depression?'].value_counts()}")
