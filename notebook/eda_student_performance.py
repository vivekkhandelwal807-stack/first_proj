"""
EDA - Student Performance Indicator
=====================================
Life cycle of Machine Learning Project:
- Understanding the Problem Statement
- Data Collection
- Data Checks
- Exploratory Data Analysis
- Data Pre-Processing
- Model Training
- Choose best model

Problem Statement:
This project understands how the student's performance (test scores) is affected
by other variables such as Gender, Ethnicity, Parental level of education,
Lunch and Test preparation course.

Dataset Source: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
The data consists of 8 columns and 1000 rows.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('data/raw.csv')

print("Top 5 Records:")
print(df.head())

print("\nShape of dataset:", df.shape)


# ─────────────────────────────────────────────
# 2. DATA CHECKS
# ─────────────────────────────────────────────

# 2.1 Check Missing Values
print("\nMissing Values:")
print(df.isna().sum())

# 2.2 Check Duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# 2.3 Check Data Types
print("\nData Info:")
df.info()

# 2.4 Unique values per column
print("\nUnique values per column:")
print(df.nunique())

# 2.5 Statistics
print("\nDataset Statistics:")
print(df.describe())

# 2.6 Explore categorical columns
print("\nCategories in 'gender':", df['gender'].unique())
print("Categories in 'race_ethnicity':", df['race_ethnicity'].unique())
print("Categories in 'parental_level_of_education':", df['parental_level_of_education'].unique())
print("Categories in 'lunch':", df['lunch'].unique())
print("Categories in 'test_preparation_course':", df['test_preparation_course'].unique())

# Define numerical & categorical columns
numeric_features = [f for f in df.columns if df[f].dtype != 'O']
categorical_features = [f for f in df.columns if df[f].dtype == 'O']
print('\nNumerical features ({})  : {}'.format(len(numeric_features), numeric_features))
print('Categorical features ({}): {}'.format(len(categorical_features), categorical_features))


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Add Total Score and Average columns
df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score'] / 3
print("\nDataset with new columns:")
print(df.head())

# Full marks & low scorers
reading_full  = df[df['reading_score'] == 100]['average'].count()
writing_full  = df[df['writing_score'] == 100]['average'].count()
math_full     = df[df['math_score'] == 100]['average'].count()

print(f'\nStudents with full marks in Maths  : {math_full}')
print(f'Students with full marks in Writing: {writing_full}')
print(f'Students with full marks in Reading: {reading_full}')

reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
math_less_20    = df[df['math_score'] <= 20]['average'].count()

print(f'\nStudents with ≤20 marks in Maths  : {math_less_20}')
print(f'Students with ≤20 marks in Writing: {writing_less_20}')
print(f'Students with ≤20 marks in Reading: {reading_less_20}')
# Insight: Students performed worst in Maths; best in Reading.


# ─────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────

# 4.1 Average Score Distribution (Histogram & KDE)
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df, x='average', bins=30, kde=True, color='g')
plt.subplot(122)
sns.histplot(data=df, x='average', kde=True, hue='gender')
plt.suptitle('Average Score Distribution')
plt.tight_layout()
plt.savefig('reports/avg_score_distribution.png')
plt.show()

# Total Score Distribution
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df, x='total score', bins=30, kde=True, color='g')
plt.subplot(122)
sns.histplot(data=df, x='total score', kde=True, hue='gender')
plt.suptitle('Total Score Distribution by Gender')
plt.tight_layout()
plt.savefig('reports/total_score_distribution.png')
plt.show()
# Insight: Female students tend to perform better than male students.

# 4.2 Lunch effect on Average Score
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
sns.histplot(data=df, x='average', kde=True, hue='lunch')
plt.subplot(142)
sns.histplot(data=df[df['gender'] == 'female'], x='average', kde=True, hue='lunch')
plt.subplot(143)
sns.histplot(data=df[df['gender'] == 'male'], x='average', kde=True, hue='lunch')
plt.suptitle('Effect of Lunch on Scores')
plt.tight_layout()
plt.savefig('reports/lunch_effect.png')
plt.show()
# Insight: Standard lunch helps perform well regardless of gender.

# 4.3 Parental Education effect on Average Score
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
ax = sns.histplot(data=df, x='average', kde=True, hue='parental_level_of_education')
plt.subplot(142)
ax = sns.histplot(data=df[df['gender'] == 'male'], x='average', kde=True, hue='parental_level_of_education')
plt.subplot(143)
ax = sns.histplot(data=df[df['gender'] == 'female'], x='average', kde=True, hue='parental_level_of_education')
plt.suptitle('Effect of Parental Education on Scores')
plt.tight_layout()
plt.savefig('reports/parental_education_effect.png')
plt.show()
# Insight: Associate's/master's degree parents → male child performs well; no clear effect for female.

# 4.4 Race/Ethnicity effect on Average Score
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
ax = sns.histplot(data=df, x='average', kde=True, hue='race_ethnicity')
plt.subplot(142)
ax = sns.histplot(data=df[df['gender'] == 'female'], x='average', kde=True, hue='race_ethnicity')
plt.subplot(143)
ax = sns.histplot(data=df[df['gender'] == 'male'], x='average', kde=True, hue='race_ethnicity')
plt.suptitle('Effect of Race/Ethnicity on Scores')
plt.tight_layout()
plt.savefig('reports/race_ethnicity_effect.png')
plt.show()
# Insight: Group A and Group B tend to perform poorly regardless of gender.

# 4.5 Violin plots – Score distributions per subject
plt.figure(figsize=(18, 8))
plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math_score', data=df, color='red', linewidth=3)
plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading_score', data=df, color='green', linewidth=3)
plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing_score', data=df, color='blue', linewidth=3)
plt.suptitle('Score Distribution per Subject')
plt.tight_layout()
plt.savefig('reports/violin_scores.png')
plt.show()
# Insight: Most students score 60-80 in Maths; 50-80 in Reading & Writing.

# 4.6 Pie plots – Multivariate overview
plt.rcParams['figure.figsize'] = (30, 12)

plt.subplot(1, 5, 1)
size = df['gender'].value_counts()
plt.pie(size, colors=['red', 'green'], labels=['Female', 'Male'], autopct='.%2f%%')
plt.title('Gender', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 2)
size = df['race_ethnicity'].value_counts()
labels = size.index
plt.pie(size, labels=labels, autopct='.%2f%%')
plt.title('Race/Ethnicity', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 3)
size = df['lunch'].value_counts()
plt.pie(size, colors=['red', 'green'], labels=['Standard', 'Free'], autopct='.%2f%%')
plt.title('Lunch', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 4)
size = df['test_preparation_course'].value_counts()
plt.pie(size, colors=['red', 'green'], labels=['None', 'Completed'], autopct='.%2f%%')
plt.title('Test Course', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 5)
size = df['parental_level_of_education'].value_counts()
plt.pie(size, labels=size.index, autopct='.%2f%%')
plt.title('Parental Education', fontsize=20)
plt.axis('off')

plt.tight_layout()
plt.savefig('reports/pie_charts.png')
plt.show()

# 4.7 Gender analysis
f, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(x=df['gender'], data=df, palette='bright', ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=20)
plt.pie(x=df['gender'].value_counts(), labels=['Male', 'Female'],
        explode=[0, 0.1], autopct='%1.1f%%', shadow=True, colors=['#ff4d4d', '#ff8000'])
plt.suptitle('Gender Distribution')
plt.tight_layout()
plt.savefig('reports/gender_distribution.png')
plt.show()
# Insight: Females 518 (48%), Males 482 (52%)

# Gender vs Average & Math score
gender_group = df.groupby('gender').mean()
print("\nGender Group Means:\n", gender_group)

plt.figure(figsize=(10, 8))
X = ['Total Average', 'Math Average']
female_scores = [gender_group['average'][0], gender_group['math_score'][0]]
male_scores   = [gender_group['average'][1], gender_group['math_score'][1]]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, male_scores,   0.4, label='Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label='Female')
plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total Average vs Math Average by Gender", fontweight='bold')
plt.legend()
plt.savefig('reports/gender_vs_scores.png')
plt.show()
# Insight: Females have better overall score; Males score higher in Maths.

# 4.8 Race/Ethnicity analysis
f, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(x=df['race_ethnicity'], data=df, palette='bright', ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=20)
plt.pie(x=df['race_ethnicity'].value_counts(),
        labels=df['race_ethnicity'].value_counts().index,
        explode=[0.1, 0, 0, 0, 0], autopct='%1.1f%%', shadow=True)
plt.suptitle('Race/Ethnicity Distribution')
plt.tight_layout()
plt.savefig('reports/race_distribution.png')
plt.show()
# Insight: Most students from Group C/D; fewest from Group A.

# Race/Ethnicity vs Subject scores
Group_data2 = df.groupby('race_ethnicity')
f, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.barplot(x=Group_data2['math_score'].mean().index, y=Group_data2['math_score'].mean().values,
            palette='mako', ax=ax[0])
ax[0].set_title('Math Score', color='#005ce6', size=20)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=15)

sns.barplot(x=Group_data2['reading_score'].mean().index, y=Group_data2['reading_score'].mean().values,
            palette='flare', ax=ax[1])
ax[1].set_title('Reading Score', color='#005ce6', size=20)
for container in ax[1].containers:
    ax[1].bar_label(container, color='black', size=15)

sns.barplot(x=Group_data2['writing_score'].mean().index, y=Group_data2['writing_score'].mean().values,
            palette='coolwarm', ax=ax[2])
ax[2].set_title('Writing Score', color='#005ce6', size=20)
for container in ax[2].containers:
    ax[2].bar_label(container, color='black', size=15)

plt.suptitle('Race/Ethnicity vs Subject Scores')
plt.tight_layout()
plt.savefig('reports/race_vs_scores.png')
plt.show()
# Insight: Group E highest; Group A lowest across all subjects.

# 4.9 Parental Education analysis
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')
sns.countplot(x='parental_level_of_education', data=df, palette='Blues')
plt.title('Comparison of Parental Education', fontsize=20)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('reports/parental_education_count.png')
plt.show()
# Insight: Largest group of parents from "some college".

df.groupby('parental_level_of_education').agg('mean').plot(kind='barh', figsize=(10, 10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Parental Education vs Mean Scores')
plt.tight_layout()
plt.savefig('reports/parental_education_vs_scores.png')
plt.show()
# Insight: Master's/Bachelor's degree parents → higher student scores.

# 4.10 Lunch type analysis
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('seaborn-talk')
sns.countplot(x='lunch', data=df, palette='PuBu')
plt.title('Comparison of Lunch Types', fontsize=20)
plt.xlabel('Type of Lunch')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('reports/lunch_count.png')
plt.show()
# Insight: More students served Standard lunch than free/reduced.

f, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.countplot(x='parental_level_of_education', data=df, palette='bright',
              hue='test_preparation_course', saturation=0.95, ax=ax[0])
ax[0].set_title('Students vs Test Preparation Course', color='black', size=25)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=20)

sns.countplot(x='parental_level_of_education', data=df, palette='bright',
              hue='lunch', saturation=0.95, ax=ax[1])
for container in ax[1].containers:
    ax[1].bar_label(container, color='black', size=20)

plt.suptitle('Parental Education vs Test Prep & Lunch')
plt.tight_layout()
plt.savefig('reports/parental_education_vs_prep_lunch.png')
plt.show()
# Insight: Standard lunch students perform better.

# 4.11 Test Preparation Course effect
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.barplot(x=df['lunch'], y=df['math_score'], hue=df['test_preparation_course'])
plt.subplot(2, 2, 2)
sns.barplot(x=df['lunch'], y=df['reading_score'], hue=df['test_preparation_course'])
plt.subplot(2, 2, 3)
sns.barplot(x=df['lunch'], y=df['writing_score'], hue=df['test_preparation_course'])
plt.suptitle('Test Prep Course Effect on Scores')
plt.tight_layout()
plt.savefig('reports/test_prep_effect.png')
plt.show()
# Insight: Students who completed test prep course score higher in all three subjects.

# 4.12 Outlier Detection – Boxplots
plt.subplots(1, 4, figsize=(16, 5))
plt.subplot(141)
sns.boxplot(y=df['math_score'],    color='skyblue')
plt.subplot(142)
sns.boxplot(y=df['reading_score'], color='hotpink')
plt.subplot(143)
sns.boxplot(y=df['writing_score'], color='yellow')
plt.subplot(144)
sns.boxplot(y=df['average'],       color='lightgreen')
plt.suptitle('Outlier Detection – Boxplots')
plt.tight_layout()
plt.savefig('reports/boxplots_outliers.png')
plt.show()

# 4.13 Pairplot – Multivariate Analysis
sns.pairplot(df, hue='gender')
plt.suptitle('Pairplot – All Scores by Gender', y=1.02)
plt.savefig('reports/pairplot.png')
plt.show()
# Insight: All scores increase linearly with each other.


# ─────────────────────────────────────────────
# 5. CONCLUSIONS
# ─────────────────────────────────────────────
print("""
=== CONCLUSIONS ===
- Student's Performance is related with lunch, race, and parental level of education.
- Females lead in pass percentage and are top-scorers overall.
- Standard lunch significantly improves performance.
- Students from Group A and Group B tend to perform poorly.
- Completing the test preparation course is beneficial.
- All three subject scores are highly correlated with each other.
""")
