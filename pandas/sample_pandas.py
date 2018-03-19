import pandas as pd
import numpy as np

s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)

datas = pd.date_range('20160101', periods=6)

print(datas)

df = pd.DataFrame(np.random.randn(), index=datas, columns=['a', 'b', 'c', 'd'])
print(df)

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

df2 = pd.DataFrame({
                   'A': [11],
                    'B': [12],
                    'C': [13]
                   })

print(df2)
# 输出列标
print(df2.index)

# 输出数据类型
print(df2.dtypes)

# 输出行标
print(df2.columns)

# 输出值
print(df2.values)

# 描述(数值类型)
print(df2.describe)

# 数组转置
print(df2.T)

# 排序(对行，倒序)
print(df2.sort_index(axis=1, ascending=False))

# 排序(对值)
print(df2.sort_values(by='A'))

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape(6,4), index=dates, columns=['A', 'B', 'C', 'D'])

print(df)

# 选择行项目
print(df.A, df['A'])


# 选择列项目
print(df['20130103':'20130105'])
print(df[0:3])


# select by label:loc
print(df.loc['20130101'])

# select by position : iloc
print(df.iloc[3:5, 1:3])
print(df.iloc[[1,2,5], 2:4])

# select by boolean indexing
print(df[df.A > 4])

print(df.A)

# --------------------------------

# 改变值
df.iloc[2, 2] = 111
print(df)

df.loc['20130101', 'A'] = 222
print(df)

df.A[df.A > 10] = 0
print(df)


# 添加空列

df['E'] = np.nan
print(df)

df['F'] = pd.Series([1, 2, 3, 4, 5, 6], index = pd.date_range('20130101', periods=6))

print(df)
df.loc['20130101':'20130104','E'] = 2
print(df)

# 处理不完整的数据
# 除去缺失的数据
print(df.dropna(axis=0, how='all'))

# 填上缺失的数据
print(df.fillna(value=0))

# 检查是否有缺失数据
print(np.any(df.isnull() == True))

# -----------------------

df.to_csv('123.csv')


# --------------------

# 合并df
# concatenating
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4))*2, columns=['a', 'b', 'c', 'd'])


# 上下合并
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print(res)

# join, ['inner', 'outer'],相同部分合并
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])

res = pd.concat([df1, df2], join='inner', ignore_index=True)
print(res)

# 左右合并(df1 的索引)
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
print(res)

# append 向下加数据
df1 = pd.DataFrame(np.ones((3,4))*0, index=[1, 2, 3], columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.ones((3,4))*1, index=[1, 2, 3], columns=['A', 'B', 'C', 'D'])
df3 = pd.DataFrame(np.ones((3,4))*2, index=[1, 2, 3], columns=['A', 'B', 'C', 'D'])
res = df1.append(df2, ignore_index=True)
res = df1.append([df2, df3], ignore_index=True)
print(res)

s = pd.Series([1, 2, 3, 4], index=['A', 'B', 'C', 'D'])
print(s)

res = df1.append(s, ignore_index=True)
print(res)


# ---------------------------------------
left = pd.DataFrame({
                    'key': ['k0', 'k0', 'k2', 'k3'],
                    'key2': ['k0', 'k1', 'k2', 'k3'],
                    'A': ['a0', 'a1', 'a2', 'a3'],
                    'B': ['b0', 'b1', 'b2', 'b3']
                    })
right = pd.DataFrame({
                    'key': ['k0', 'k1', 'k1', 'k3'],
                    'key2': ['k0', 'k0', 'k0', 'k3'],
                    'C': ['c0', 'c1', 'c2', 'c3'],
                    'D': ['d0', 'd1', 'd2', 'd3']
                    })
# 合并DF， merge
res = pd.merge(left, right, on='key')
print(res)

#  默认how='inner'
# how = ['left', 'right', 'inner', 'outer']
res = pd.merge(left, right, on=['key', 'key2'], how='left', indicator=True)
res = pd.merge(left, right, on=['key', 'key2'], how='left', indicator='indicator_colunmn')
print(res)


# index合并
res = pd.merge(left, right, left_index=True, right_index=True, how='outer', indicator=True)
print(res)

# handle overlapping


#------------------
#pandas plot
import matplotlib.pyplot as plt

data = pd.Series(np.random.randn(1000), index=np.arange(1000))
print(data)

data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list('ABCD'))
data = data.cumsum()

#plot methods:
#条形图'bar','box', 'kde', hist, area, scatter, hexbin, pie
ax1 = data.plot.scatter(x='A', y='B', color='DarkBlue', label="Class 1")
ax2 = data.plot.scatter(x='A', y='D', color='r', label="Class 3", ax=ax1)
data.plot.scatter(x='A', y='C', color='g', label="Class 2", ax=ax2)
data.plot()
plt.show()
