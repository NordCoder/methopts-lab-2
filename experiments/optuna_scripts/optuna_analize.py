import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

custom_bfgs_df = pd.read_csv('optuna_report/CustomBfgs_optimization_results.csv')
custom_gradient_descent_df = pd.read_csv('optuna_report/CustomGradientDescentOptimizer_optimization_results.csv')
newton_line_search_df = pd.read_csv('optuna_report/NewtonLineSearch_optimization_results.csv')
scipy_bfgs_df = pd.read_csv('optuna_report/SciPyBFGS_optimization_results.csv')
scipy_cg_df = pd.read_csv('optuna_report/SciPyCG_optimization_results.csv')
scipy_nelder_mead_df = pd.read_csv('optuna_report/SciPyNelderMead_optimization_results.csv')

dfs = [
    ("CustomBfgs", custom_bfgs_df),
    ("CustomGradientDescent", custom_gradient_descent_df),
    ("NewtonLineSearch", newton_line_search_df),
    ("SciPyBFGS", scipy_bfgs_df),
    ("SciPyCG", scipy_cg_df),
    ("SciPyNelderMead", scipy_nelder_mead_df)
]

combined_df = pd.concat([df.assign(Method=method) for method, df in dfs], ignore_index=True)

mean_values = combined_df.groupby('Method')[['tol', 'max_iter']].mean()
std_values = combined_df.groupby('Method')[['tol', 'max_iter']].std()

print("Средние значения гиперпараметров:")
print(mean_values)

print("\nСтандартные отклонения гиперпараметров:")
print(std_values)

# Гистограмма для 'tol' по каждому методу
plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='tol', data=combined_df)
plt.title('Сравнение гиперпараметра "tol" для разных методов')
plt.show()

# Гистограмма для 'max_iter' по каждому методу
plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='max_iter', data=combined_df)
plt.title('Сравнение гиперпараметра "max_iter" для разных методов')
plt.show()

# Сравниваем минимальные значения функции для каждого метода
best_results = combined_df.loc[combined_df.groupby('Method')['f'].idxmin()]

print("Лучшие результаты оптимизации для каждого метода:")
print(best_results[['Method', 'Function', 'f', 'tol', 'max_iter']])

# Корреляционная матрица для гиперпараметров
correlation_matrix = combined_df[['tol', 'max_iter']].corr()
print("Корреляционная матрица между гиперпараметрами:")
print(correlation_matrix)
