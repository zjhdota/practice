from scipy.special import comb, perm
a = sum(comb(48, i) for i in range(1, 50))
print(a)
