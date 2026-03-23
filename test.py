from openai import OpenAI
from pydantic import BaseModel, Field
import json
import os

MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

with open('input.txt', 'r', encoding='utf-8') as f:
    content = f.read()

with open('prompt.txt', 'r', encoding='utf-8') as f:
    prompt = f.read()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("ROUTER_API_KEY")
)


class Task(BaseModel):
    name: str = Field(description="Название задачи")
    desc: str = Field(default="", max_length=250,
                      description="Краткое описание (не более 200 токенов)")
    prio: int = Field(ge=1, le=5, description="Приоритет от 1 до 5")
    time: int = Field(gt=0, description="Время в часах")


completion = client.chat.completions.parse(
    model=MODEL,
    messages=[
        {
            "role": "system", "content": prompt
        },
        {
            "role": "user", "content": content
        },
    ],
    temperature=0.3,
    response_format=Task,
)

print((completion.choices[0].message.parsed))
