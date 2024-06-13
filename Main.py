import System as sy
import seaborn as sns
import matplotlib.pyplot as plt 

print('hola')
ret = sy.SystemFUN()

print(ret.head())
sns.lmplot(
    data=ret, x="Test size", y="Score", row="Normalization method", col="Feature selection method", hue='Model name',
    palette="muted", ci=None,
    height=4, scatter_kws={"s": 50, "alpha": 1}
)
plt.show()