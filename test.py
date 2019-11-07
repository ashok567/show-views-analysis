import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/data.csv')


def days_count(date):
    before_start = dt.date(2017, 2, 28)
    days = date.date() - before_start
    days = str(days).split()[0]
    return days


def weekdays(days):
    weekdays = (days+3) % 7  # star date is wed
    if weekdays == 0:
        weekdays = 7
    return weekdays


# Date Processing
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = df['Date'].apply(days_count).astype(int)
df['Week_key'] = df['Days'].apply(weekdays)
df['Weekend'] = df['Week_key'].apply(lambda x: 1 if x in [1, 7] else 0)

print(df.head())

# Scaling
df['Ad_impression_thousands'] = round(df['Ad_impression']/1000).astype(int)

# EDA
fig = plt.figure()
ax = fig.add_subplot(111)
ax = sns.lineplot(df['Days'], df['Viewership'])
plt.show()
plt.cla()
# Viewerships are good on most of the weekends
ax = sns.scatterplot(x='Days', y='Viewership', data=df, hue='Weekend')
plt.show()
plt.cla()
# Viewership has picked up in presence of a character but again fall down in its absence
ax = sns.scatterplot(x='Days', y='Viewership', data=df, hue='CharacterA')
plt.xlabel('Days')
plt.ylabel('Viewership')
plt.legend()
plt.show()
# Weekend and Ad Impression are correlated with Viewership
ax = sns.heatmap(df.corr(), annot=True)
plt.show()
