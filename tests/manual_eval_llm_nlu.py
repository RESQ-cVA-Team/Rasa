"""Manual, live-LLM evaluation for the experimental LLM-only NLU pipeline
(src/components/llm_nlu_classifier.py). Not a unit test -- needs a trained
model and a real configured LLM provider, so it isn't run by the test
suite. Trains and runs:

    ./scripts/layer_rasa_projects.sh src/core src/locales/en/US
    python tests/manual_eval_llm_nlu.py

Prints intent/confidence/entities for a small held-out batch. See the branch
this lives on for what a run of this actually turned up.
"""
import asyncio
import glob
import json

from rasa.core.agent import Agent

MODEL = sorted(glob.glob("models/*.tar.gz"))[-1]

TEST_MESSAGES = [
    "hi there",
    "show me door to needle for men",
    "what chart types are available?",
    "plot thrombolysis rate as a bar chart grouped by stroke type",
    "show pre-stroke mrs for women younger than 45",
    "what's the weather like today",
    "goodbye",
    "who are you",
    "show me a line chart of hospital stay",
    "compare perfusion core volume for male patients",
]

async def main():
    agent = Agent.load(MODEL)
    for text in TEST_MESSAGES:
        result = await agent.parse_message(text)
        intent = result.get("intent", {})
        entities = result.get("entities", [])
        ent_summary = [
            {"entity": e.get("entity"), "value": e.get("value"), "role": e.get("role")}
            for e in entities
        ]
        print(json.dumps({
            "text": text,
            "intent": intent.get("name"),
            "confidence": round(intent.get("confidence", 0), 3),
            "entities": ent_summary,
        }))

asyncio.run(main())
