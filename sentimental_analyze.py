import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

def main():
    model = AutoModelForSequenceClassification.from_pretrained("KernAI/stock-news-distilbert")
    tokenizer = AutoTokenizer.from_pretrained("KernAI/stock-news-distilbert")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    data = pd.read_json("./news.json")
    
    results = []
    
    for summary, datetime in zip(data['summary'], data['datetime']):
        res = classifier(summary)
        result = res[0]
        results.append({"datetime": datetime, "label": result['label'], "score": result['score']})
        
    df = pd.DataFrame(results)
    df.to_csv("./stock/classifier_results.csv", index=False)
        
if __name__ == "__main__":
    main()