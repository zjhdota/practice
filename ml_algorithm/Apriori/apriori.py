import pandas as pd


df = pd.read_csv('data.csv', delimiter=',')
df = pd.get_dummies(df)
df = (df==1)

min_support = 0.1
min_confidence = 0.2

support = df.sum() / len(df)
column = list(support[support > min_support].index)
result = pd.DataFrame(index=['support', 'confidence'])
ms = '--'
while len(column) > 1:
    print('-------------------')
    column = [ sorted(i.split(ms)) for i in column]
    l = len(column[0])
    r = []
    for i in range(len(column)):
        for j in range(i,len(column)):
            if column[i][:l-1] == column[j][:l-1] and column[i][l-1] != column[j][l-1]:
                r.append(column[i][:l-1]+sorted([column[j][l-1],column[i][l-1]]))
    column = r

    # 计算连接后的支持度
    sf = lambda x: df[list(x)].prod(axis=1, numeric_only = True)
    df_2 = pd.DataFrame(list(map(sf, column)), index = [ms.join(i) for i in column]).T
    support2 = df_2[[ms.join(i) for i in column]].sum()/len(df)
    support = support.append(support2)

    #新一轮支持度筛选
    column = list(support2[support2 > min_support].index)

    # 获取元素组合
    column2 = []
    for i in column:
        i = i.split(ms)
        for j in range(len(i)):
                column2.append(i[:j]+i[j+1:]+i[j:j+1])

    cofidence_series = pd.Series(index=[ms.join(i) for i in column2]) #定义置信度序列

    for i in column2: #计算置信度序列
        cofidence_series[ms.join(i)] = support[ms.join(sorted(i))]/support[ms.join(i[:len(i)-1])]

    for i in cofidence_series[cofidence_series > min_confidence].index: #置信度筛选
        result[i] = 0.0
        result[i]['confidence'] = cofidence_series[i]
        result[i]['support'] = support[ms.join(sorted(i.split(ms)))]

result = result.T.sort_values(['confidence','support'], ascending = False) #结果整理，输出
print(result)

