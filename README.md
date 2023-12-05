EASY 1
Given a string s consisting of words and spaces, return the length of the last word in the string
def lastword(w):
    lw = w.strip().split()
    if not lw:
        return 0
    return len(lw[-1])

w = input();
print(lastword(w))  

easy2
Given an integer array nums where the elements are sorted in ascending order, convert it to a 
height-balanced binary search tree.
 
class TreeNode(object):
    def _init_(self, val):
        self.val = val
        self.left = None
        self.right = None

def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid + 1:])
    return root
nums = [-10, -3, 0, 5, 9]
bst = sortedArrayToBST(nums)
print(bst.val) 


Easy 3
Given an integer numRows, return the first numRows of Pascal's triangle.

def ptriangle(rows):
    tri = [[1]]
    if rows == 0:
        return []
    for i in range(1, rows):
        prev_row = tri[-1]
        new_row = [1]
        
        for j in range(1, len(prev_row)):
            new_row.append(prev_row[j - 1] + prev_row[j])
        
        new_row.append(1)
        tri.append(new_row)
    return tri

r1 = 5
print(ptriangle(r1)) 
r2 = 1
print(ptriangle(r2)) 


medium 1
Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def lowestCommonAncestor(root, p, q):
    if not root:
        return None

    if p.val > root.val and q.val > root.val:
        return lowestCommonAncestor(root.right, p, q)
    elif p.val < root.val and q.val < root.val:
        return lowestCommonAncestor(root.left, p, q)
    else:
        return root

root = TreeNode(6)
root.left = TreeNode(2)
root.right = TreeNode(8)
root.left.left = TreeNode(0)
root.left.right = TreeNode(4)
root.right.left = TreeNode(7)
root.right.right = TreeNode(9)
root.left.right.left = TreeNode(3)
root.left.right.right = TreeNode(5)

p = TreeNode(2)
q = TreeNode(8)

lca = lowestCommonAncestor(root, p, q)
if lca:
    print("Lowest Common Ancestor of", p.val, "and", q.val, "is", lca.val)
else:
    print("Nodes not found or invalid BST.")



Medium 2 

Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.
from collections import Counter
def majority_elements(nums):
    n = len(nums)
    th = n // 3
    counts = Counter(nums)
    result = [num for num, count in counts.items() if count > th]
    return result
n1 = [3, 2, 3]
print(majority_elements(n1))
n2 = [1]
print(majority_elements(n2)) 
n3 = [1, 2]
print(majority_elements(n3))


medium 3

def maximalSquare(matrix):
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    max_side = 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == '1':
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                max_side = max(max_side, dp[i][j])

    return max_side * max_side

# Example usage:
matrix1 = [["0","1"],["1","0"]]
matrix2 = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]
print("Maximal square in matrix1:", maximalSquare(matrix1))
print("Maximal square in matrix2:", maximalSquare(matrix2))

hard1

from collections import deque

def maxSlidingWindow(nums, k):
    if not nums:
        return []

    result = []
    window = deque()

    for i in range(len(nums)):
        while window and window[0] < i - k + 1:
            window.popleft()

        while window and nums[i] > nums[window[-1]]:
            window.pop()

        window.append(i)

        if i >= k - 1:
            result.append(nums[window[0]])

    return result


nums1 = [1,3,-1,-3,5,3,6,7]
nums2 = [1]
result1 = maxSlidingWindow(nums1, 3)
result2 = maxSlidingWindow(nums2, 1)
print(result2)
print(result1)


hard 2
You are given a string s. You can convert s to a 
palindrome by adding characters in front of it.
Return the shortest palindrome you can find by performing this transformation

def shortest_palindrome(s):
    i = 0
    for j in range(len(s) - 1, -1, -1):
        if s[i] == s[j]:
            i += 1
    if i == len(s):
        return s
    suffix = s[i:]
    prefix = suffix[::-1]
    return prefix + s

s=input()
result = shortest_palindrome(s)
print(f"The shortest palindrome for '{s}' is: {result}")


hard 3
Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.
n = int(input())
count = 0
for i in range(1, n + 1):
    count += str(i).count('1')
print(f'Total number of digit 1 in integers up to {n}: {count}')
