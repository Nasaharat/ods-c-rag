"""OpenAI wrapper: embeddings and chat."""

import os

from openai import OpenAI


class LLMClient:
    """Handles embedding and chat calls to the OpenAI API."""

    def __init__(self, chat_model="gpt-4o-mini",
                 embedding_model="text-embedding-3-small"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key, timeout=30.0)
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed(self, texts):
        """Return one embedding vector per input string."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=texts
            )
        except Exception as error:
            raise RuntimeError(f"Embedding request failed: {error}")
        return [item.embedding for item in response.data]

    def chat(self, system_prompt, user_prompt):
        """Return the model's reply to a system and user prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as error:
            raise RuntimeError(f"Chat request failed: {error}")
        return response.choices[0].message.content
