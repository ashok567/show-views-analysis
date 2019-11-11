import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# After exxploring data it is clear that show views are affected by Weekends, Characters and Ad Imressions
print("Without  Ad Impression....")
x1 = df[['Visitors', 'Weekend', 'CharacterA']]
y = df['Viewership']

lm1 = LinearRegression()
lm1.fit(x1, y)

prediction1 = lm1.predict(x1)

mse1 = mean_squared_error(y, prediction1)
r_squared1 = r2_score(y, prediction1)

print('Mean_Squared_Error :', mse1)
print('r_square_value :', r_squared1)

fig = plt.figure()
plt.plot(df['Days'], y, c='blue')
plt.plot(df['Days'], prediction1, c='red')
fig.suptitle('Actual and Predicted - Model1')
plt.xlabel('Days')
plt.ylabel('Viewership')
plt.show()
plt.cla()

print("\n")
print("Adding  Ad Impression....")
x2 = df[['Visitors', 'Weekend', 'CharacterA', 'Ad_impression_thousands']]

lm2 = LinearRegression()
lm2.fit(x2, y)

prediction2 = lm2.predict(x2)

mse2 = mean_squared_error(y, prediction2)
r_squared2 = r2_score(y, prediction2)

print('Mean_Squared_Error :', mse2)
print('r_square_value :', r_squared2)

plt.plot(df['Days'], y, c='blue')
plt.plot(df['Days'], prediction2, c='red')
fig.suptitle('Actual and Predicted - Model2')
plt.xlabel('Days')
plt.ylabel('Viewership')
plt.show()
