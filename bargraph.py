import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
means_decisiontree = (80,14,85,14)
means_naivebayesain = (64,23,76,23)
means_knn = (76,35,64,35)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 1

rects1 = plt.bar(index, means_decisiontree, bar_width,
                 alpha=opacity,
                 color='b',
                 label='DecisionTree')

rects2 = plt.bar(index + bar_width, means_naivebayesain, bar_width,
                 alpha=opacity,
                 color='g',
                 label='NaiveBayesian')
index1 = index + bar_width

rects3 = plt.bar(index1 + bar_width, means_knn, bar_width,
                 alpha=opacity,
                 color='r',
                 label='knn')

plt.xlabel('Algorithms')
plt.ylabel('Scores')
plt.title('Comparision')
plt.xticks(index + bar_width, ('Accuracy', 'Classification error', 'Sensitivity', 'Specificity'))
plt.legend()

plt.tight_layout()
plt.show()