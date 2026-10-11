import json
import os
import re
import urllib.request
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID = {"happy", "sad", "neutral"}

# ---------- Fallback: tight rule-based classifier ----------
HAPPY = {
    "love", "loved", "loves", "like", "likes", "great", "good", "excellent",
    "amazing", "awesome", "fantastic", "wonderful", "happy", "glad", "delighted",
    "joy", "joyful", "excited", "thrilled", "pleased", "perfect", "best",
    "brilliant", "enjoy", "enjoyed", "beautiful", "nice", "proud", "grateful",
    "thankful", "fun", "yay", "congratulations", "congrats", "celebrate",
    "celebrating", "smile", "smiling", "laugh", "laughing", "won", "win",
    "success", "lovely", "terrific", "impressed", "delicious", "relieved",
    "hooray", "adore", "fabulous", "incredible", "superb", "thank", "thanks",
}
SAD = {
    "sad", "terrible", "awful", "bad", "hate", "hated", "horrible", "worst",
    "disappointed", "disappointing", "depressed", "unhappy", "miserable", "cry",
    "crying", "cried", "lonely", "heartbroken", "upset", "angry", "furious",
    "annoyed", "failed", "failure", "lost", "grief", "grieving", "sorrow",
    "pain", "painful", "hurt", "hurts", "tragic", "gloomy", "hopeless",
    "worried", "anxious", "stressed", "exhausted", "sick", "disaster", "ugly",
    "useless", "regret", "devastated", "dreadful", "pathetic", "frustrated",
    "unfortunately", "died", "death", "tears", "heartbreaking", "despair",
    "suffering", "betrayed", "rejected", "miserable", "awful", "depressing",
}
NEG = {"not", "no", "never", "cannot", "cant", "dont", "didnt", "isnt",
       "wasnt", "wont", "hardly", "barely", "without"}


def rule_classify(sentence: str) -> str:
    tokens = re.findall(r"[a-z']+", sentence.lower().replace("’", "'"))
    score = 0
    for i, t in enumerate(tokens):
        v = 1 if t in HAPPY else -1 if t in SAD else 0
        if not v:
            continue
        window = tokens[max(0, i - 3):i]
        neg = any(w in NEG or w.endswith("n't") for w in window)
        score += -v if neg else v
    return "happy" if score > 0 else "sad" if score < 0 else "neutral"


# ---------- LLM classifier via AIPipe ----------
def llm_classify(sentences: List[str]):
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        return None
    payload = {
        "model": "openai/gpt-4o-mini",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the sentiment of each sentence as exactly one of: "
                    "happy, sad, neutral. Use happy for positive emotion or good "
                    "news, sad for negative emotion, complaints or bad news, and "
                    "neutral for factual statements with no emotion. Reply only "
                    'with JSON: {"labels": ["happy","sad",...]} with one label '
                    "per sentence, in the same order."
                ),
            },
            {"role": "user", "content": json.dumps(sentences)},
        ],
    }
    req = urllib.request.Request(
        "https://aipipe.org/openrouter/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        labels = json.loads(data["choices"][0]["message"]["content"])["labels"]
        labels = [str(l).strip().lower() for l in labels]
        if len(labels) == len(sentences) and all(l in VALID for l in labels):
            return labels
    except Exception:
        pass
    return None


class SentimentRequest(BaseModel):
    sentences: List[str]


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    labels = llm_classify(req.sentences) or [rule_classify(s) for s in req.sentences]
    return {
        "results": [
            {"sentence": s, "sentiment": l} for s, l in zip(req.sentences, labels)
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))