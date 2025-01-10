import numpy as np
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import pandas as pd

# ВАРИАНТ 1
#
#
data = '{-216.1; -219.9; -212.3; -221.4; NA; -179.5; -189.9; -219; -178.1; NA; -189; NA; -181.3; -323.725; -203.3; -191.3; -171.6; NA; -203.5; -207.6; -230.1; -212.2; -197.4; -186.4; -195.9; -185.7; NA; -203.3; -203; NA; -206.8; -253.7; NA; -216.4; -189.3; -225.6; -166.8; -226.5; -174; -192.9; -194.1; -189.8; NA; -212; -170.2; -205.7; -185.7; -194.7; -205.5; -210.6; -218.3; -225.9; -199.1; NA; -184.4; -202.7; NA; -193.7; -185.9; -217; NA; -209.3; -220.2; NA; -175; -210.2; -210.2; -221.1; -183.1; -178.4; -236.4; -220.1; -234.2; NA; -197; -163.6; -203.4; NA; -230.7; -180.3; -178.6; NA; -229.1; -193.2; -202.6; -242.3; -220; -200.6; -202.2; -308.95; -173.5; -230.3; NA; -172.3; -198.7; -175.7; -207; -210.2; -223.5; -200.6; -217.6; -220.3; -229.1; -216.5; -208.1; -228.9; -230.9; NA; -190.3; -196; -204.3; -206.3; -198.3; -293.975; -208.3; -209.2; NA; -214.1; -180.4; -205.3; -191.6; -253; -206.5; -184.6; -208.4; -205; -221.3; -224.8; -224.6; -189.3; NA; -241.2; -218.6; -215.7; -200.4; -220.5; -222.7; -204.8; -219.5; -201.5; -174.3; -214.4; -212.2; -230.4; -207.9; -151; -205.3; NA; NA; -233.1; -183.3; -224; 3.52499999999978; -181.3; -239.2; -207.6; -204.7; -181; -184.7; -172.5; -174.5; NA; -212; -229.6; -190.8; -245.2; -212.9; -207; NA; -208.6; NA; -179.6; -196.2; -198.4; NA; -202.1; -205.1; -227.6; -206.7; -202.5; NA; -209.1; -224.7; -197.2; -210.4; -229.9; NA; -191.7; NA; -191.4; -256.4; -195.8; -190; -219.7; -210.3; -165.5; -230; -204; -182.8; -186.1; -179.7; -170.1; -223.6; -206.5; -176.2; -222.4; -185.4; -194.3; -169.6; -237.6; -233.3; -205.6; -214.9; NA; -193.4; -225.4; -220.6; -183.4; -201.2; -245.3; NA; -183.4; -215.8; -208.4; -170.2; -149.5; -217.7; -212.1; -207.9; -215.5; -206.7; -222.5; -198.9; -246.7; -217.7; -241.6; NA; -211.2; -213.4; -201.4; NA; -55.9750000000002; NA; -193.4; -237.9; -26.2250000000002; NA; -212.3; -177.3; NA; -171.3; -165.7; -217.3; -206.8; NA; -227.7; NA; NA; -189; -182.8; -182.1; -203.9; NA; -209.2; -223.2; -227.7; -204.6; -180.2; -209.7; -202.4; -199.3; -175.4; NA; -211.3; -176.5; NA; -212.2; -185.4; -206.4; -206; -183.9; -189.4; -226.4; -216.8; -222.5; -172.3; -185.3; -226; -176.2; -198.3; -155.2; -194.6; -192.7; -257.1; -218.1; -221.3; -224.4; -298.55; -236.7; -210.6}.'
data = data[1:-2].split(sep =  '; ')
data_grap = data.copy()
n = len(data)
data = [i for i in data if (i!=' NA') and (i!='NA') ]
data = pd.Series([float(i.replace(',','.')) for i in data if (i!=' NA') or (i!='NA')])
n_without = len(data)
print('Объем без выбросов и NA ',n_without)

print('Минимальное значение в вариационном ряду',min(data))
print('Максимальное значение в вариационном ряду',max(data))
print('Размах выборки',round(max(data)-min(data),3))

Q1 = np.quantile(data, 0.25)
print('Значение первой квартили (Q1)',round(Q1,3))
Q2 = np.quantile(data, 0.5)
print('Значение медианы (Q2)',round(Q2,3))
Q3 = np.quantile(data, 0.75)
print('Значение третьей квартили (Q3)',round(Q3,3))
R = Q3-Q1
print('Квартильный размах',round(R,3))
mean = data.mean()
print('Среднее выборочное значение',round(mean,3))

std_corr = data.std(ddof=1)
print('Стандартное отклонение (S) корень из дисп.в (исправленной)',round(std_corr ,3))
var_corr = data.var(ddof=1)
print('Исправленная дисперсия ',round(var_corr,3))
kurt = sts.kurtosis(data, bias=False)
print('Эксцесс ',round(sts.kurtosis(data, bias=False),3))
skew = sts.skew(data, bias=False)
print('Коэффициент асимметрии',round(skew,3))
error = std_corr/n_without**0.5
print('Ошибка выборки',round(error,3))

# Тут посмотри какая у тебя гамма в условии, чтоб был верный ответ
gamma = 0.9
interv = sts.t.interval(gamma,n-1,  mean,  std_corr/np.sqrt(n_without))
round(interv[0],3),round(interv[1],3)

gamma = 0.9
chi2_gamma1 = sts.chi2.ppf((1-gamma)/2, n_without-1)
chi2_gamma2 = sts.chi2.ppf((1+gamma)/2, n_without-1)
round((n_without-1)*var_corr/chi2_gamma2,3), round((n_without-1)*var_corr/chi2_gamma1,3)


x_stat_max = Q3+1.5*R
print('Верхняя граница нормы (Xst_max)', round(x_stat_max,3))
x_stat_min =  Q1-1.5*R
print('Нижняя граница нормы (Xst_min)', round(x_stat_min,3))
print('Количество выбросов ниже нижней нормы',len(data[data < x_stat_min]))
print('Количество выбросов выше верхней нормы',len(data[data > x_stat_max]))

data = pd.Series([float(i.replace(',', '.')) for i in data_grap if i != 'NA' ])

plt.hist(data, bins=25, edgecolor='black')
plt.title('Гистограмма c выбросами')
plt.show()

plt.boxplot(data, vert=True, patch_artist=True,  showmeans=True)
plt.title('Диаграмма "Ящик с усиками" с выбросами')
plt.show()

data = pd.Series([ i for i in data if i!=np.nan])
data = data[(data<x_stat_max) & (data>x_stat_min)]

plt.hist(data, bins=10, edgecolor='black')
plt.title('Гистограмма без выбросов и NA ')
plt.show()

plt.boxplot(data, vert=True, patch_artist=True,  showmeans=True)
plt.title('Диаграмма "Ящик с усиками" без выбросов и NA')
plt.show()

#ВАРИАНТ 2
#
#
# Исходная выборка
data = ["Wen", "Mon", "Mon", "Th", "Wen", "Sun", "Wen", "Fr", "Sun", "Wen", "Wen", "Wen", "Th", "NA", "Mon", "NA", "Fr", "Mon", "Wen", "Wen", "NA", "Fr", "Mon", "Fr", "Th", "NA", "Sun", "NA", "Fr", "Fr", "Mon", "Fr", "Wen", "Wen", "NA", "Fr", "Sat", "Mon", "Th", "Fr", "Mon", "Fr", "Sun", "Sun", "Sun", "Mon", "Mon", "Th", "Th", "Mon", "Mon", "Fr", "Th", "Sun", "Mon", "Fr", "Wen", "Mon", "Th", "Sat", "Wen", "Mon", "Wen", "Wen", "NA", "Sun", "Mon", "Fr", "Mon", "Wen", "Th", "Sat", "Fr", "NA", "Wen", "Wen", "Sat", "Sat", "Th", "Mon", "Sun", "Fr", "Wen", "Th", "Wen", "Fr", "Mon", "Fr", "Sat", "Fr", "Fr", "Mon", "Wen", "Sun", "Mon", "Wen", "Sat", "Mon", "Wen", "Fr", "Fr", "Mon", "Fr", "Th", "NA", "Fr", "Sun", "Sat", "Sun", "Mon", "Mon", "Wen", "Fr", "Wen", "Mon", "Fr", "Mon", "Mon", "Sun", "Mon", "Mon", "Sat", "Th", "NA", "Sun", "Sun", "Fr", "Wen", "Sun", "Mon", "Mon", "NA", "Sun", "NA", "Wen", "Wen", "Wen", "Mon", "Wen", "Sat", "NA", "Th", "Sun", "Wen", "Th", "Sun", "Sat", "Wen", "Mon", "Wen", "NA", "Th", "Mon", "Wen", "Th", "NA", "Mon", "Fr", "Wen", "NA", "Fr", "Th", "Sun", "NA", "Wen", "Mon", "Fr", "Wen", "Th", "Fr", "Wen", "Th", "Fr", "Sat", "Sat", "Sun", "Sat", "Wen", "Sun", "Wen", "Mon", "NA", "Sat", "Fr", "Sun", "Sat", "Mon", "Sun", "Fr", "Sat", "Wen", "Wen", "Sat", "Sat", "NA", "Th", "Mon", "Mon", "Fr", "Wen", "Mon", "Wen", "Fr", "Mon", "Sat", "Th", "Fr", "Fr", "Wen", "Wen", "Sat", "Wen", "Mon", "Sun", "Sat", "Sat", "Mon", "Mon", "Mon", "Wen", "Sun", "Sun", "Sun", "Mon", "Sun", "Sat", "NA", "Sat", "Fr", "Wen", "Mon", "Sun", "Fr", "Sat", "Sat", "NA", "Sat", "NA", "Mon", "Fr", "Wen", "Wen", "Fr", "Sun", "Fr", "Mon", "Wen", "Sun", "Mon", "Mon", "Sat", "Wen", "Mon", "Fr", "NA", "Wen", "Mon", "Sat", "Wen", "Sat", "Wen", "Mon", "Fr", "Sun", "Mon", "Fr", "Wen", "Fr", "Mon", "Mon", "Fr", "Wen", "Sat", "NA", "Sun", "NA", "Wen", "Fr", "Sun", "Mon", "Sat", "Fr", "Mon", "Mon", "Mon", "Th", "Mon", "Th", "Mon", "Mon", "Sat", "NA", "Sun", "Fr", "Sun", "Mon", "Sun", "Sun", "Sat", "Wen", "Sun", "NA", "Th", "NA", "Mon", "Mon", "Fr", "Th", "Mon", "Sat", "Th", "Mon", "Mon", "Fr", "Th", "Wen", "NA", "Sat", "Mon", "Fr"]
# Очистка выборки от пропусков "NA"
cleaned_data = [item for item in data if item != "NA"]
# Вывод очищенной выборки
print(cleaned_data)

unique_answers = set(cleaned_data)

# Количество различных вариантов ответов
num_unique_answers = len(unique_answers)

# Вывод результата
print("Количество различных вариантов ответов:", num_unique_answers)
size_of_cleaned_data = len(cleaned_data)

# Вывод результата
print("Объем очищенной выборки:", size_of_cleaned_data)
count_na = data.count("NA")

# Вывод результата
print("Количество пропущенных данных 'NA':", count_na)

count_mon = cleaned_data.count("Mon")
fraction_mon = count_mon / size_of_cleaned_data
print("Доля респондентов, которые дали ответ 'Mon':", fraction_mon)

p_hat = count_mon / size_of_cleaned_data

# Критическое значение для 0.99-доверительного интервала
z = 2.576

# Правая граница 0.99-доверительного интервала
right_bound = p_hat + z * math.sqrt(p_hat * (1 - p_hat) / size_of_cleaned_data)

# Вывод результата
print("Правая граница 0.99-доверительного интервала для истинной доли ответов 'Mon':", right_bound)

left_bound = p_hat - z * math.sqrt(p_hat * (1 - p_hat) / size_of_cleaned_data)

# Вывод результата
print("Левая граница 0.99-доверительного интервала для истинной доли ответов 'Mon':", left_bound)

# Объем очищенной выборки
n = len(cleaned_data)

# Количество категорий (без "NA")
k = 6

# Ожидаемое количество ответов в каждой категории
E = n / k

# Подсчет наблюдаемых частот
observed_frequencies = {
    "Wen": cleaned_data.count("Wen"),
    "Mon": cleaned_data.count("Mon"),
    "Th": cleaned_data.count("Th"),
    "Fr": cleaned_data.count("Fr"),
    "Sat": cleaned_data.count("Sat"),
    "Sun": cleaned_data.count("Sun")
}

# Вычисление статистики хи-квадрат
chi_square = sum((observed_frequencies[category] - E)**2 / E for category in observed_frequencies)

# Вывод результата
print("Наблюдаемое значение статистики хи-квадрат:", chi_square)

# Критическое значение для 0.01 уровня значимости и 5 степеней свободы
critical_value = chi2.ppf(0.99, 5)

# Вывод критического значения
print("Критическое значение статистики хи-квадрат:", critical_value)


#ВАРИАНТ 3
#
#
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Исходные данные (в виде строки)
data_str = 'F; F; Child; Child; NA; M; F; F; Child; NA; M; NA; Child; NA; NA; NA; Child; Child; F; Child; Child; F; F; M; M; Child; M; M; Child; M; F; NA; F; M; M; NA; M; F; Child; Child; NA; M; Child; Child; Child; Child; F; M; Child; Child; F; M; Child; Child; F; M; F; M; F; M; F; M; M; Child; M; NA; Child; Child; NA; NA; Child; Child; Child; F; Child; F; F; F; F; Child; Child; M; M; Child; F; Child; NA; Child; M; F; Child; M; Child; M; Child; M; M; Child; Child; Child; NA; Child; NA; F; F; Child; F; Child; M; Child; Child; F; Child; F; F; Child; Child; F; Child; M; Child; Child; F; Child; Child; NA; Child; M; M; F; Child; NA; M; F; M; M; F; Child; F; NA; F; M; NA; F; NA; Child; M; M; F; F; F; M; Child; M; F; M; M; M; M; F; F; Child; M; NA; F; NA; NA; M; Child; M; Child; M; M; F; F; F; M; M; M; F; M; M; M; F; NA; F; Child; Child; F; F; M; F; NA; Child; M; M; M; F; F; F; M; M; NA; Child; M; F; Child; NA; M; Child; Child; M; Child; F; F; F; Child; NA; NA; F; M; M; F; Child; M; NA; F; NA; Child; F; F; Child; M; NA; Child; F; M; NA; Child; NA; NA; NA; Child; Child; M; M; Child; F; NA; NA; NA; M; M; NA; M; Child; M; M; Child; F; Child; Child; Child; F; F; M; M; M; F; F; Child; Child; F; Child; Child; F; M; Child; Child; Child; Child; M; F; Child; NA; M; Child; M; NA; F; Child; M; F; F; F; NA; Child; Child; M; Child; F; F; M; NA; F; M; Child; M; F; Child; M; M; F; M; Child; Child; F; M; F; M'

# Преобразуем строку в список, удалив разделители и пустые элементы
data_list = [x.strip() for x in data_str.split(';') if x.strip() != "NA"]

# Выводим очищенные данные
print(f"Очищенные данные: {data_list[:10]}...")  # Показываем первые 10 элементов для проверки

# Вычисляем количество уникальных ответов в очищенной выборке
unique_answers = set(data_list)
print(f"Количество уникальных вариантов ответов: {len(unique_answers)}")

# Вычисляем объем очищенной выборки
cleaned_data_size = len(data_list)
print(f"Объем очищенной выборки: {cleaned_data_size}")

# Пример: Подсчет, сколько раз встречается каждый вариант
from collections import Counter
answer_counts = Counter(data_list)
print(f"Частота каждого варианта ответа: {answer_counts}")

cleaned_data = [response for response in data_list if response != "NA"]

# Получение уникальных значений в очищенной выборке
unique_responses = set(cleaned_data)
cleaned_data_volume = len(cleaned_data)
na_count = data_str.split(';').count(' NA')

# Вывод количества различных вариантов
print(f"Количество различных вариантов ответов: {len(unique_responses)}")
print(f"Объем очищенной выборки: {cleaned_data_volume}")
print(f"Количество пропущенных данных 'NA' в исходной выборке: {na_count}")

mon_count = data_str.split(';').count(' F')

# Подсчитываем общее количество ответов, исключая "NA"
total_count = len([x for x in data_str.split(';') if x != ' NA'])

# Вычисляем долю респондентов, которые выбрали "Mon"
mon_ratio = mon_count / total_count

# Выводим результат
print(f"Доля респондентов, которые дали ответ 'F': {mon_ratio:.4f}")

import scipy.stats as stats

# Количество категорий (возможно, ответов типа "Mon", "Tue", "Wed", "NA", и другие)
categories = ['F', 'M', 'Child']

# Количество степеней свободы: количество категорий - 1
df = len(categories) - 1  # 7 категорий, поэтому df = 6

# Уровень значимости (alpha)
alpha = 0.1

# Критическое значение для хи-квадрат распределения с 6 степенями свободы и уровнем значимости 0.01
critical_value = stats.chi2.ppf(1 - alpha, df)

print(f"Критическое значение статистики хи-квадрат для уровня значимости 0.1 и {df} степеней свободы: {critical_value:.4f}")

import math

# Исходная выборка

# Подсчет количества ответов "Mon"
count_mon = cleaned_data.count("F")

# Объем очищенной выборки
size_of_cleaned_data = len(cleaned_data)

# Выборочная доля ответов "Mon"
p_hat = count_mon / size_of_cleaned_data

# Критическое значение для 0.99-доверительного интервала
z = 4.6052

# Правая граница 0.99-доверительного интервала
right_bound = p_hat + z * math.sqrt(p_hat * (1 - p_hat) / size_of_cleaned_data)

# Вывод результата
print("Правая граница 0.99-доверительного интервала для истинной доли ответов 'Mon':", right_bound)

import math

# Исходная выборка
# Очистка выборки от пропусков "NA"

# Подсчет количества ответов "Mon"
count_mon = cleaned_data.count("F")

# Объем очищенной выборки
size_of_cleaned_data = len(cleaned_data)

# Выборочная доля ответов "Mon"
p_hat = count_mon / size_of_cleaned_data

# Критическое значение для 0.99-доверительного интервала
z = 4.6052

# Левая граница 0.99-доверительного интервала
left_bound = p_hat - z * math.sqrt(p_hat * (1 - p_hat) / size_of_cleaned_data)

# Вывод результата
print("Левая граница 0.99-доверительного интервала для истинной доли ответов 'Mon':", left_bound)


# Подсчитываем количество ответов для каждой категории (день недели)
category_counts = pd.Series(cleaned_data).value_counts()

# Общее количество респондентов после удаления "NA"
total_responses = len(cleaned_data)

# Ожидаемое количество для каждой категории при равномерном распределении
expected_count = total_responses / len(category_counts)

# Вычисляем наблюдаемое значение Хи-квадрат
chi_square_statistic = ((category_counts - expected_count) ** 2 / expected_count).sum()

# Выводим результат
print(f"Наблюдаемое значение хи-квадрат: {chi_square_statistic:.4f}")


# Проверка, отвергаем ли гипотезу
if chi_square_statistic > critical_value:
    result = 1  # Отвергаем гипотезу
else:
    result = 0  # Не отвергаем гипотезу

print(f"Есть ли основания отвергнуть гипотезу о равновероятном распределении? Введите: {result}")


plt.hist(cleaned_data)
plt.show()

# Исходные данные
data = [
    (215.83, 177.5), (219.21, 256.6), (186.22, 214.1), (186.42, 214.5), (232.31, 216.7),
    (250.72, 'NA'), (222.69, 222.3), (195.03, 192.9), (251.54, 261.3), (235.56, 200.2),
    (243.2, 'NA'), (230.44, 170.6), ('NA', 'NA'), (232.74, 227.2), (188.78, 204.3),
    (235.33, 219.8), (207.53, 242.7), (258.02, 212.1), ('NA', 220.4), (219.02, 164.5),
    (241.03, 208.9), (226.71, 228.1), (257.26, 206.8), (168.37, 232.4), (174.24, 225.8),
    (195.9, 168.1), (215.01, 191.3), (172.09, 195.8), (249.97, 223.8), (219.97, 253.8),
    (201.34, 177.0), (181.6, 198.0), (201.0, 188.3), (201.75, 154.3), (193.25, 202.1),
    (234.34, 212.9), (222.78, 164.3), (165.67, 170.8), (196.18, 198.6), (241.45, 235.0),
    ('NA', 213.5), (187.2, 182.5), ('NA', 217.6), (170.55, 239.0), (213.26, 224.4),
    (203.35, 185.5), (213.12, 276.1), (201.82, 200.0), (194.01, 'NA'), (157.24, 191.7),
    (223.46, 188.8), (221.68, 232.7), (209.33, 192.9), (152.45, 189.9), (192.22, 223.4),
    (208.33, 181.8), (253.81, 212.2), ('NA', 215.3), (244.21, 170.7), (193.45, 203.7),
    (223.12, 206.4), ('NA', 218.3), (244.15, 187.3), (181.17, 221.5), (216.27, 238.5),
    (202.03, 241.3), (271.87, 216.0), (152.51, 244.4), (222.22, 214.1), ('NA', 260.3),
    (218.05, 242.4), (193.42, 269.8), (192.26, 184.7), (176.34, 200.5), ('NA', 216.0),
    (201.41, 228.2), (236.32, 'NA'), (179.99, 207.0), (208.24, 204.4), (196.84, 229.1),
    (169.63, 181.7), ('NA', 221.0), (142.93, 260.9), (263.33, 208.8), (200.63, 'NA'),
    (220.23, 222.1), (199.53, 'NA'), (218.37, 238.7), (211.98, 204.4), (175.46, 208.5),
    (222.54, 231.2), (249.94, 216.3), (175.46, 230.5), (185.44, 256.7), (225.5, 223.7),
    (209.13, 200.4), (239.82, 217.5), (197.82, 247.0), (206.29, 200.9), (182.11, 231.2),
    (232.23, 211.7), (219.33, 262.1), (189.37, 235.0), (226.23, 234.8), (201.34, 231.2),
    (198.61, 208.4), (245.77, 'NA'), (190.04, 210.8), (255.04, 'NA'), (178.03, 170.5),
    (219.83, 216.3), (165.29, 206.7), (159.89, 239.6), ('NA', 196.7), (158.96, 218.7),
    (196.23, 182.6), (203.53, 215.6), (218.15, 237.5), (177.68, 'NA'), (198.89, 'NA'),
    (220.24, 246.7), ('NA', 202.8), (217.28, 184.6), ('NA', 223.3), (192.3, 201.7),
    (231.5, 230.3), (265.79, 207.1), (196.15, 183.3), (214.2, 243.9), (211.26, 236.2),
    (182.27, 164.7), (229.09, 226.6), (207.8, 282.9), (215.11, 221.6), (187.19, 208.0),
    (226.49, 208.1), (163.74, 195.5), (173.93, 187.6), (225.97, 228.5), (203.08, 154.8),
    (215.06, 243.3), (212.4, 217.7), (167.35, 187.0), (171.18, 224.0),
]

# Функция для замены 'NA' на None
def replace_na(data):
    return [(x if x != 'NA' else None, y if y != 'NA' else None) for x, y in data]

# Применяем замену
data_clean = replace_na(data)

# Преобразуем в pandas DataFrame для удобной работы с данными
df = pd.DataFrame(data_clean, columns=["X", "Y"])

# Удаляем строки с None
df_cleaned = df.dropna()

# Рассчитаем статистику
mean_x = df_cleaned["X"].mean()
mean_y = df_cleaned["Y"].mean()
median_x = df_cleaned["X"].median()
median_y = df_cleaned["Y"].median()

print(f"Среднее X: {mean_x}")
print(f"Среднее Y: {mean_y}")
print(f"Медиана X: {median_x}")
print(f"Медиана Y: {median_y}")


def replace_na(data):
    return [(x if x != 'NA' else None, y if y != 'NA' else None) for x, y in data]

# Применяем замену
data_clean = replace_na(data)

# Преобразуем в pandas DataFrame для удобной работы с данными
df = pd.DataFrame(data_clean, columns=["X", "Y"])

# Удаляем строки с None
df_cleaned = df.dropna()

# Рассчитаем выборочный коэффициент корреляции Пирсона между X и Y
correlation = df_cleaned["X"].corr(df_cleaned["Y"])

print(f"Выборочный коэффициент корреляции Пирсона: {correlation}")

import pandas as pd

# Данные выборки
data = [
    (-178.6, -183.8241); (None, None); (-183.3, -123.9528); (None, -114.3199); (-122.2, -160.5017); (None, -156.0367); (-182.3, -172.576); (-171.8, -172.804); (-116.5, -203.4581); (-195.5, -159.3408); (-215, -113.0763); (-153.3, None); (-171, -155.2601); (-167.6, None); (-132.1, -174.4958); (-204.3, -135.3174); (-131, -144.6971); (-158.4, -193.2912); (-124.4, -188.0632); (None, -187.6458); (-164.5, -128.9331); (-200.2, -148.895); (-171.3, -178.3126); (-141.1, -181.4645); (-146.3, -178.3586); (-132.1, -163.2404); (-177.9, None); (-143.7, -198.43); (-164.7, -159.1087); (-132.8, -155.5847); (-183.7, -120.6001); (-169.7, -159.3242); (-174.9, -191.2892); (-191.5, -169.3699); (-161, -173.8607); (-172.8, -194.5698); (-150.8, -130.2788); (-144, -119.7459); (-163.1, -158.9308); (-150.9, -139.4356); (-174.2, -167.8212); (-204.7, -172.9995); (-183.8, -146.1889); (-195.6, -198.3558); (-131.7, -173.5364); (-110.9, -144.6917); (-164.6, -144.006); (-168.1, -162.3919); (-141.9, -207.7032); (-210.6, -101.3587); (-138.8, -164.0083); (-216.1, -139.8154); (-186.5, -149.4878); (-162.1, -167.3145); (-124.9, -255.6124); (-150.5, -127.5702); (-214.8, -165.0173); (-169.4, -150.3155); (-207.9, -118.415); (-109.9, -171.3923); (-188.2, -181.3529); (-171.6, -156.1474); (-98.1, -154.9463); (-155.3, -127.973); (-130.4, None); (None, -132.072);(-181.6, -179.9056); (-198.1, -153.5997); (-210.3, -205.0083); (None, -200.9914); (-144.5, -45.4229); (-105.4, -175.0903); (-167.3, -172.2534); (None, -204.3272); (-173, -144.2191); (-146.8, -126.4174); (None, -184.727); (-154.4, -126.1183); (-160.4, -140.085); (None, -169.6621); (-110.9, -188.6073); (None, -120.2488); (-211.4, -205.5225); (-144.4, None); (-184.2, -166.8992); (-140.2, -137.7707); (-197.8, -183.7244); (-155.8, -142.0952); (-146.5, -201.0108); (-172.5, -211.5637); (-178.6, None); (-167.2, None); (-162, -178.7688); (-183.2, -116.5135); (-173.6, -139.9831); (-109.4, None); (-154.4, -193.7977); (-194.4, None); (-146.9, -201.5813); (-213.5, -210.5296); (-159.5, -124.8299); (-140.8, -177.3902); (None, -154.6959); (-200.4, -163.7636); (-187.2, -135.0145); (-155.3, -157.92); (-164.7, -161.2); (-190.4, -167.8168); (-160.5, -145.7185); (-171, None); (-115.4, -134.3015); (-176.1, -133.0603); (-161.5, -165.6095); (-195.6, -130.8721); (-161.6, -205.8107); (-153.3, None); (-162.4, -157.6832); (-182.4, -161.9254); (-185.8, None); (None, -163.7555); (-179.6, -166.2891); (-153.1, None); (None, -177.9315); (-190.3, -154.1313); (-176.6, -155.5416); (-155.7, -151.5819); (-167.9, None); (-162.4, -149.4022); (-181.6, -167.3182); (-192.1, -187.0777); (-113.5, -172.4841); (-163.4, -149.5614); (-162, -159.942); (-188.4, -199.7505); (-126.4, -138.1175); (-166.5, -136.8558); (-183, None); (-125.1, -156.776); (-169, -191.612); (-154.7, -146.2857); (-145.6, -130.5467); (-125.4, -134.1148); (-185.4, None); (-180.6, -167.1542); (-161.3, -140.6649); (-171.6, -147.0023); (-202.9, -178.6365); (-148.6, -173.3995); (-126.7, -190.0803); (-185, -161.6488)
]

# Создание DataFrame
df = pd.DataFrame(data, columns=['A', 'B'])

df_cleaned = df.dropna()

df.head()

df = pd.DataFrame(data, columns=['A', 'B'])

# Удаление строк с пропущенными данными
df_cleaned = df.dropna()

# Вычисление выборочного коэффициента корреляции Пирсона
corr_XY = df_cleaned['A'].corr(df_cleaned['B'])

# Вывод результата
print(f"Выборочный коэффициент корреляции Пирсона между X и Y: {corr_XY:.4f}")

# Проверка гипотезы о равенстве средних значений показателей фирм
# Альтернативная гипотеза: среднее значение показателя больше у второй фирмы
t_stat, p_value = ttest_ind(df_cleaned['A'], df_cleaned['B'], equal_var=False, alternative='less')
result = 1 if p_value <= alpha else 0
# Вывод результата
print(f"T-статистика: {t_stat:.4f}")
print(f"P-значение: {p_value:.4f}")
print(f"Результат проверки гипотезы: {result}")

t_stat, p_value = ttest_ind(df_cleaned['A'], df_cleaned['B'], equal_var=False, alternative='less')

# Уровень значимости
alpha = 0.05

# Проверка условия
result = 1 if p_value <= alpha else 0

# Вывод результата
print(f"P-значение: {p_value}")
print(f"Результат проверки гипотезы: {result}")

from scipy.stats import levene

stat, p_value = levene(df_cleaned['A'], df_cleaned['B'], center='mean')

# Вывод результата
print(f"Статистика Левена: {stat:.4f}")
print(f"P-значение: {p_value:.4f}")

alpha = 0.1

# Проверка условия
result = 1 if p_value <= alpha else 0

# Вывод результата
print(f"P-значение: {p_value}")
print(f"Результат проверки гипотезы: {result}")
