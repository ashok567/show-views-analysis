import pandas as pd
import datetime as dt
df = pd.read_csv('data/data.csv')


def days(date):
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
df['date_key'] = df['Date'].apply(days).astype(int)
df['week_key'] = df['date_key'].apply(weekdays)

# Scaling
df['Ad_impression_thousands'] = round(df['Ad_impression']/1000).astype(int)