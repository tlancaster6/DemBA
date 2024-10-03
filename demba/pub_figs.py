import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


prequant_map5s = pd.DataFrame(
    [
        [2007, 0.733798086643219],
        [1600, 0.7333183288574219],
        [800, 0.7050431370735168],
        [400, 0.6861478090286255],
        [200, 0.6454454660415649],
        [100, 0.5862843990325928],
        [50,	0.497409850358963],
        [25,	0.39085280895233154]
    ],
    columns=['number of training samples', 'prequantization mAP@.5']
)

fig, ax = plt.subplots(figsize=(6, 3))
sns.lineplot(prequant_map5s, x='number of training samples', y='prequantization mAP@.5', ax=ax, marker='o')
ax.set(title='EfficientDet-Lite0 Fish Detection Accuracy vs.\nNumber of Training Samples')
fig.tight_layout()
fig.savefig('randn_analysis.pdf')
plt.close(fig)

