# -*- encoding=utf-8 -*-
L = [3, 1, 4, 2, 20, 15, 21, 14]

# # 后面的数和前面已经排序好的相比较
# def insert_sort(L):
#     for i in range(1, len(L)):
#         key = L[i]
#         j = i-1
#         while j >= 0 and key < L[j]:
#             L[j+1], L[j] = L[j], L[j+1]
#             j = j - 1
#     return L

# print(insert_sort(L))

# # 每一轮都找出最大的一个元素
# # 每次相比元素 len(L)-i
# def bubble_sort(L):
#     for i in range(1, len(L)):
#         for j in range(0, len(L)-i):
#             if L[j] > L[j+1]:
#                 L[j], L[j+1] = L[j+1], L[j]
#     return L
# print(bubble_sort(L))

# # [3, 1, 2, 4]
# # pythonic 函数调用次数增加
# def quick_sort(L):
#     if not L:
#         return []
#     else:
#         key = L[0]
#         less = [i for i in L if key > i]
#         more = [i for i in L[1:] if key <= i]
#     return quick_sort(less) + [key] + quick_sort(more)

# print(quick_sort(L))

# def merge_sort(L):
#     mid = int(len(L)/2)
#     if len(L) <= 1:
#         return L
#     else:
#         #print(L[:mid])
#         # print(L[mid:])
#         # print("---------")
#         left = merge_sort(L[:mid])
#         right = merge_sort(L[mid:])
#         # print(left, right)
#         result = []
#         i, j = 0, 0
#         while i < len(left) and j < len(right):
#             if left[i] <= right[j]:
#                 result.append(left[i])
#                 i += 1
#             else:
#                 result.append(right[j])
#                 j += 1
#         result.extend(left[i:])
#         result.extend(right[j:])
#         return result

# print(merge_sort(L))


def maxheep(L, start, end):
    root = start
    while True:
        child = root * 2 + 1
        if child > end: break
        if child + 1 <= end and L[child] < L[child+1]:
            child = child + 1
        if L[root] < L[child]:
            L[root], L[child] = L[child], L[root]
            root = child
        else:
            break
def heap_sort(L):
    n = len(L)
    first = int(len(L) / 2 - 1)
    for start in range(first, -1, -1):
        maxheep(L, start, n-1)
    for end in range(n-1, 0, -1):
        L[end], L[0] = L[0], L[end]
        maxheep(L, 0, end-1)
    return L

print(heap_sort(L))
