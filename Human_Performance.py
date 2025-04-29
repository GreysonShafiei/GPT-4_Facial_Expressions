import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset file, select the agreement and emotional display
df = pd.read_csv("WSEFEP - norms & FACS - Arkusz1.csv", encoding="utf-8")

# Average the agreement for each emotion
agg = (df.groupby("Display", as_index=False)["Agreement (%)"].mean())
emotion_order = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
agg = agg.set_index("Display").loc[emotion_order].reset_index()
agg["Agreement (%)"] = agg["Agreement (%)"] / 100.0

plt.figure(figsize=(10, 5))
bars = plt.bar(agg["Display"], agg["Agreement (%)"])
plt.ylim(0, 1)
plt.xlabel("Emotion")
plt.ylabel("Agreement (%)")
plt.title("Per-Emotion Agreement")
plt.tight_layout()

for bar, value in zip(bars, agg["Agreement (%)"]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 0.01,
             f"{value:.2f}",
             ha='center', va='bottom', fontsize=9)

plt.savefig("human_agreement_by_emotion.png", dpi=300)
plt.show()
