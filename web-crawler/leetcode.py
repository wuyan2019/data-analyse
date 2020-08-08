# # class ListNode:
# #     def __init__(self, x):
# #         self.val = x
# #         self.next = None
# #
# #
# # def addTwoNumbers(l1: ListNode, l2: ListNode):
# #     # 这里一开始不做l1、l2非空判断，题意表明非空链表
# #     # 记录是否需要增加新节点，或在链表下一个节点是否需要加1，同时记录链表同级节点的和
# #     carry = 0
# #     # 这里的执行顺序是res = ListNode(0), pre = res
# #     res = pre = ListNode(0)
# #     # 判断l1、l2、carry是否有值，carry有值的话需要增加新节点，或在链表下一个节点需要加1
# #     while l1 or l2 or carry:
# #         # 判断l1是否有值
# #         if l1:
# #             carry += l1.val
# #             l1 = l1.next
# #         # 判断l2是否有值
# #         if l2:
# #             carry += l2.val
# #             l2 = l2.next
# #         # carry有同级节点的和
# #         # divmod返回商与余数的元组，拆包为carry和val
# #         # carry有值的话需要增加新节点，或在链表下一个节点需要加1,在循环中会用到
# #         carry, val = divmod(carry, 10)
# #         # 新建链表节点
# #         # 这里是n.next = ListNode(val), n = n.next()
# #         pre.next = pre = ListNode(val)
# #     # res等价于pre，res.val=0，所以返回res.next
# #     return res.next
# #
# #
# # if __name__ == '__main__':
# #     # 创建对象Solution
# #     # 定义l1链表
# #     l1 = ListNode(2)
# #     l1.next = l11 = ListNode(4)
# #     l11.next = l12 = ListNode(5)
# #     # 定义l2链表
# #     l2 = ListNode(5)
# #     l2.next = l21 = ListNode(6)
# #     l21.next = l22 = ListNode(4)
# #     # 获取返回值的链表
# #     res = addTwoNumbers(l1, l2)
# #     # 循环遍历出来
# #     while res:
# #         print(res.val)
# #         res = res.next
#
#
# def lengthOfLongestSubstring(s: str):
#     # maxlen = 0
#     # memo = dict()
#     # begin, end = 0, 0
#     # n = len(s)
#     # while end < n:
#     #     last = memo.get(s[end])
#     #     memo[s[end]] = end
#     #     if last is not None:
#     #         maxlen = max(maxlen, end - begin)
#     #         begin = max(begin, last + 1)
#     #     end += 1
#     # maxlen = max(maxlen, end - begin)
#     # return maxlen
#
#     max_number = 0
#     number = 0
#     test = ""
#     for i in s:
#         if i not in test:
#             test += i
#             number += 1
#         else:
#             if number >= max_number:
#                 max_number = number
#             index = test.index(i)
#             test = test[index + 1:] + i
#             number = len(test)
#         if number >= max_number:
#             max_number = number
#     return max_number
#
#
# b = lengthOfLongestSubstring("pwwkew")
#


# import json
# n = '{"a": 1, "b": 2}'
# try:
#     nn = json.loads(n)
#     res = dict()
#     for i in nn:
#         res[nn[i]] = i
#     print(res)
# except Exception as e:
#     print('输入错误')


# def findMedianSortedArrays(nums1, nums2):
#     m, n = len(nums1), len(nums2)
#     i, j, k = 0, 0, 0
#     r1, r2 = 0, 0
#
#     while k <= (m + n) // 2:
#
#         if j >= n or (i < m and nums1[i] < nums2[j]):
#             r1, r2 = r2, nums1[i]
#             i += 1
#             k += 1
#         else:
#             r1, r2 = r2, nums2[j]
#             j += 1
#             k += 1
#     return float(r2) if (m + n) % 2 != 0 else (r1 + r2) / 2


# def reverse(x):
#     if x == 0:
#         return x
#     negative = False
#     if x < 0:
#         negative = True
#         x = abs(x)
#     x_string = str(x)
#     x_reverse = x_string[::-1]
#     i = 0
#     for index in x_reverse:
#         if index == '0':
#             i += 1
#         else:
#             break
#     if not negative:
#         x = int(x_reverse[i:])
#     else:
#         x = int('-' + x_reverse[i:])
#     return x if -2147483648 < x < 2147483647 else 0


def romanToInt(s):
    roman_dict = {'I': 1, 'V': 5, 'X': 10,  'L': 50, 'C': 100, 'D': 500, 'M': 1000,
                  'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
    if len(s) == '':
        return 0
    res = 0
    s_len = len(s)
    i = 0
    while i < s_len:
        if roman_dict.get(s[i:i+2]):
            res += roman_dict.get(s[i:i+2])
            i += 2
        else:
            res += roman_dict.get(s[i])
            i += 1
    return res


def intToRoman(num):
    A = ['I', 'X', 'C', 'M']
    B = ['V', 'L', 'D']
    H = ''
    for i in range(4):
        k, num = num % 10, num // 10
        print(k, num)
        if k < 4:
            H = k * A[i] + H
        elif k < 5:
            H = A[i] + B[i] + H
        elif k < 9:
            H = B[i] + (k - 5) * A[i] + H
        else:
            H = A[i] + A[i + 1] + H
    return H


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def mergeTwoLists(l1, l2):
    head = ListNode(0)
    first = head
    while l1 is not None and l2 is not None:
        if l1.val > l2.val:
            head.next = l2
            l2 = l2.next
        else:
            head.next = l1
            l1 = l1.next
        head = head.next
    if l1 is None:
        head.next = l2
    elif l2 is None:
        head.next = l1
    return first.next


def threeSumClosest(nums, target):
    nums_l = len(nums)
    if nums_l < 3:
        return []
    nums.sort()
    for i in range(nums_l - 2):
        start = i + 1
        end = nums_l - 1
        close = 99
        while start < end:
            res = nums[i] + nums[start] + nums[end]
            print(nums[i], nums[start], nums[end])
            if res < target:
                start += 1
            elif res > target:
                end -= 1
            else:
                return res
            a1 = target - close
            a2 = target - res
            if abs(a1) > abs(a2):
                close = res
    return close


nums = [-1, 2, 1, -4]
target = 1
b = threeSumClosest(nums, target)
