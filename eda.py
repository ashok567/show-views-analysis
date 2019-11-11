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
    weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    weekday = (days+3) % 7
    if weekday == 0:
        weekday = 7
    return weekday_names[weekday-1]


# Date Processing
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = df['Date'].apply(days_count).astype(int)
df['Weekdays'] = df['Days'].apply(weekdays)
df['Weekend'] = df['Weekdays'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Scaling
df['Ad_impression_thousands'] = round(df['Ad_impression']/1000).astype(int)

# EDA
fig = plt.figure()
ax = fig.add_subplot(111)
ax = sns.lineplot(df['Days'], df['Viewership'])
plt.show()
plt.cla()

# Viewerships are good on most of the weekends
ax = sns.boxplot(x='Weekdays', y='Viewership', data=df)
plt.show()
plt.cla()

# Viewership has picked up in presence of a character but again fell down in its absence
ax = sns.stripplot(x='CharacterA', y='Viewership', data=df, hue='Weekdays', jitter=True)
plt.show()
plt.cla()

# Viewership has picked up even on weekends in presence of a character
ax = sns.boxplot(x='Weekdays', y='Viewership', data=df, hue='CharacterA')
plt.show()
plt.cla()

# There is no such impact of a cricket match at any day
ax = sns.scatterplot(x="Days", y="Viewership", hue="Cricket_match", data=df)
plt.xlabel('Days')
plt.ylabel('Viewership')
plt.legend()
plt.show()


# Weekend and Ad Impression are correlated with Viewership
ax = sns.heatmap(df.corr(), annot=True)
plt.show()
