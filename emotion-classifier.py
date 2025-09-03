
from transformers import pipeline

class EmotionClassifier:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    def classify(self, text: str):
        result = self.classifier(text)[0]
        emotion = self.detect_emotion_distribution(result)
        return emotion  
    def detect_emotion_distribution(self, results, threshold=0.15):
        # Sort by score, high → low
        sorted_res = sorted(results, key=lambda x: x['score'], reverse=True)

        top1, top2 = sorted_res[0], sorted_res[1]

        if (top1['score'] - top2['score']) < threshold:
            # Mixed emotions
            return [top1['label'], top2['label']]
        else:
            # Single dominant emotion
            return [top1['label']]

if __name__ == "__main__":
    ec = EmotionClassifier()
    text = "I'm feeling confused today."
    print(ec.classify(text))