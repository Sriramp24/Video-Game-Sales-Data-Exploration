# ================================================================
# DATA SCIENCE ANALYSIS SCRIPT: VIDEO GAME SALES DATA
# ================================================================
# This script loads video game sales data, cleans it, and performs
# several analyses and visualizations, including descriptive and 
# inferential statistics, hypothesis testing, and data visualization.
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_ind,normaltest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Set Seaborn style for plots
sns.set(style="whitegrid")
df = pd.read_csv("cleaned_data.csv")
df_sample = df.copy()
print("----- Objective 1: Data Overview and Filtering by Critic Score -----")
print(df_sample.head())


df_sample['release_date'] = pd.to_datetime(
    df_sample['release_date'],
    format='%d-%m-%Y',  # or whatever your date format is
    errors='coerce'
)

print(df_sample.dtypes)

def filter_high_score(dataframe, threshold=80):
    return dataframe[dataframe['critic_score'] > threshold]

high_score_games = filter_high_score(df_sample, threshold=80)
print(high_score_games.head())

print("----- Objective 2: Categorize Games by Total Sales -----")
def categorize_sales(sales):
    if sales > 5:
        return 'High'
    elif sales > 2:
        return 'Medium'
    else:
        return 'Low'

df_sample['sales_category'] = df_sample['total_sales'].apply(categorize_sales)
print(df_sample[['total_sales', 'sales_category']].head())

print("----- Objective 3: Handle Missing Values and Additional Metrics -----")
df_sample.ffill(inplace=True)



df_sample['sales_percentage'] = (df_sample['total_sales'] / df_sample['total_sales'].sum()) * 100
print(df_sample[['total_sales', 'sales_percentage']].head())

genre_avg_score = df_sample.groupby('genre')['critic_score'].mean().reset_index()
print("Average Critic Score by Genre:")
print(genre_avg_score)

top10_sales = df_sample.sort_values(by='total_sales', ascending=False).head(10)
print("Top 10 Best-Selling Games:")
print(top10_sales[['title', 'total_sales']])

print("----- Objective 4: Visualizations -----")
plt.figure(figsize=(10, 5))
top5_publishers = df_sample.groupby('publisher')['total_sales'].sum().nlargest(5)
top5_publishers.plot(kind='bar', color='skyblue')
plt.title('Top 5 Publishers by Total Sales')
plt.xlabel('Publisher')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df_sample['critic_score'], bins=20, kde=True, color='green')
plt.title('Distribution of Critic Scores')
plt.xlabel('Critic Score')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.scatterplot(x='critic_score', y='total_sales', data=df_sample, alpha=0.5)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
region_sales = df_sample[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']].sum()
plt.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', startangle=140)
plt.title('Regional Sales Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
numeric_cols = df_sample.select_dtypes(include=np.number)
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

print("----- Objective 5: Advanced Statistical Analysis -----")

print("Summary Statistics:")
print(df_sample.describe())

plt.figure(figsize=(8, 5))
sns.boxplot(y=df_sample['total_sales'], color='violet')
plt.title('Outliers in Total Sales')
plt.tight_layout()
plt.show()

corr_value = df_sample['critic_score'].corr(df_sample['total_sales'])
print(f"Correlation between Critic Score and Total Sales: {corr_value:.2f}")

plt.figure(figsize=(10, 5))
df_sample['genre'].value_counts().plot(kind='bar', color='orange')
plt.title('Most Common Game Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

top_publishers = df_sample['publisher'].value_counts().nlargest(10).index
filtered_df = df_sample[df_sample['publisher'].isin(top_publishers)]
contingency_table = pd.crosstab(filtered_df['genre'], filtered_df['publisher'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test result: chi2 = {chi2:.2f}, p-value = {p:.5f}")

print("----- New Objective 6: Normality Test for Critic Scores -----")
stat, p_value = normaltest(df_sample['critic_score'].dropna())  
print(f"Normality Test (D’Agostino’s K^2): Statistic = {stat:.3f}, p-value = {p_value:.5f}")

if p_value > 0.05:
    print("Data is likely normally distributed.")
else:
    print("Data is likely not normally distributed.")

print("----- New Objective 7: t-Test between Action and Shooter Genres -----")

action_sales = df_sample[df_sample['genre'] == 'Action']['total_sales']
shooter_sales = df_sample[df_sample['genre'] == 'Shooter']['total_sales']
if len(action_sales) > 0 and len(shooter_sales) > 0:
    t_stat, p_val = ttest_ind(action_sales, shooter_sales, equal_var=False)
    print(f"T-Test: t-statistic = {t_stat:.3f}, p-value = {p_val:.5f}")
else:
    print("Not enough data for one or both genres for t-test.")

print("----- New Objective 8: Multicollinearity Check (VIF) -----")

numeric_data = df_sample[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']]
X = add_constant(numeric_data)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

print("----- New Objective 9: Game Release Frequency Per Year -----")

df_sample['year'] = df_sample['release_date'].dt.year
year_counts = df_sample['year'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.barplot(x=year_counts.index, y=year_counts.values, palette="Blues_d")
plt.title('Number of Game Releases Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("----- New Objective 10: A/B Testing Simulation (PS4 vs X360 Sales) -----")
ps4_sales = df_sample[df_sample['console'] == 'PS4']['total_sales']
x360_sales = df_sample[df_sample['console'] == 'X360']['total_sales']
if len(ps4_sales) > 0 and len(x360_sales) > 0:
    t_stat, p_val = ttest_ind(ps4_sales, x360_sales, equal_var=False)
    print(f"A/B Test (PS4 vs X360): t-statistic = {t_stat:.2f}, p-value = {p_val:.5f}")
else:
    print("Not enough data for one or both consoles for A/B test.")

print("----- Analysis Completed -----")
