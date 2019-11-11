import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/data.csv')


def days_count(date):
    before_start = dt.date(2017, 2, 28)
    days = date.date() - before_start
    days = str(days).split()[0]
    return days


def weekdays(days):
    weekday = (days+3) % 7  # star date is wed
    if weekday == 0:
        weekday = 7
    return weekday


# Date Processing
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = df['Date'].apply(days_count).astype(int)
df['Weekdays'] = df['Days'].apply(weekdays)
df['Weekend'] = df['Weekdays'].apply(lambda x: 1 if x in [1, 7] else 0)

# Scaling
df['Ad_impression_thousands'] = round(df['Ad_impression']/1000).astype(int)

x = df[['Weekend', 'CharacterA', 'Ad_impression_thousands']]
# x = df[['weekend', 'CharacterA', 'Visitors']]
y = df['Viewership']

lm = LinearRegression()

lm.fit(x, y)

x = sm.add_constant(x)

lm_10 = sm.OLS(y, x).fit()
# print(lm_1.summary())

prediction = lm_10.predict(x)

mse = mean_squared_error(y, prediction)
r_squared = r2_score(y, prediction)

print('Mean_Squared_Error :', mse)
print('r_square_value :', r_squared)
