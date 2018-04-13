# -*- encoding=utf-8 -*-
nums = [i for i in range(1, 100)]
def binary_search(nums, value=10):
    l = len(nums)
    mid = int(l/2)
    start = 0
    end = l
    while start <= end:
        if value > nums[mid]:
            start = mid + 1
        if value < nums[mid]:
            end = mid - 1
        if value == nums[mid]:
            return mid
        mid = int((start + end) / 2)
    return mid

print(binary_search(nums))
