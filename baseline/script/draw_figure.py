import matplotlib.pyplot as plt

# 区间和对应数量
intervals = [(0,0) , (1, 15), (16, 50), (51, 100), (101, float('inf'))]
counts = [75, 465, 126, 32, 15]

# 为每个区间创建标签
labels = [f"[{start},{int(end) if end != float('inf') else '∞'}]" for start, end in intervals]

# 画图
plt.figure(figsize=(8, 5))

# 设置柱宽：默认是 0.8，这里可以试试 0.4 更窄一些
bar_width = 0.4
plt.bar(labels, counts, width=bar_width, color='pink', edgecolor='black')

# 添加标题和标签
plt.title("Interval Distribution")
plt.xlabel("Value Range")
plt.ylabel("Count of Frame")

# 在柱子上标注数量
for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha='center')

plt.tight_layout()
plt.savefig('frame_distribution.png')