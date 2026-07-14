import os

from openai import OpenAI


def main():
    model = os.environ["OPENAI_MODEL"]
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "unused"),
    )

    models = client.models.list()
    assert [item.id for item in models.data] == [model]

    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply OK"}],
        max_tokens=2,
        temperature=0,
    )
    assert reply.choices[0].message.content is not None

    chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply OK"}],
        max_tokens=2,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    assert any(chunk.usage is not None for chunk in chunks)


if __name__ == "__main__":
    main()
