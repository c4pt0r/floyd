import os

from openai import OpenAI


def assert_integer(value, minimum):
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value >= minimum


def assert_usage(usage):
    assert usage is not None
    assert_integer(usage.prompt_tokens, 1)
    assert_integer(usage.completion_tokens, 0)
    assert_integer(usage.total_tokens, 1)
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.prompt_tokens_details is not None
    assert_integer(usage.prompt_tokens_details.cached_tokens, 0)


def main():
    model = os.environ["OPENAI_MODEL"]
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "unused"),
    )

    models = client.models.list()
    assert models.object == "list"
    assert len(models.data) == 1
    assert models.data[0].id == model
    assert models.data[0].object == "model"

    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply OK"}],
        max_tokens=2,
        temperature=0,
    )
    assert isinstance(reply.id, str) and reply.id.startswith("chatcmpl-")
    assert len(reply.id) > len("chatcmpl-")
    assert reply.object == "chat.completion"
    assert_integer(reply.created, 1)
    assert reply.model == model
    assert len(reply.choices) == 1
    assert reply.choices[0].index == 0
    assert reply.choices[0].message.role == "assistant"
    assert isinstance(reply.choices[0].message.content, str)
    assert reply.choices[0].finish_reason in ("stop", "length")
    assert_usage(reply.usage)

    chunks = list(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_tokens=2,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert chunks
    stream_id = chunks[0].id
    assert isinstance(stream_id, str) and stream_id

    role_chunks = 0
    content_chunks = 0
    terminal_chunks = 0
    usage_chunks = 0
    for index, chunk in enumerate(chunks):
        assert chunk.id == stream_id
        assert chunk.object == "chat.completion.chunk"
        assert_integer(chunk.created, 1)
        assert chunk.model == model

        if chunk.usage is not None:
            usage_chunks += 1
            assert index == len(chunks) - 1
            assert chunk.choices == []
            assert_usage(chunk.usage)
            continue

        assert len(chunk.choices) == 1
        choice = chunk.choices[0]
        assert choice.index == 0
        if choice.finish_reason is not None:
            terminal_chunks += 1
            assert index == len(chunks) - 2
            assert choice.finish_reason in ("stop", "length")
            assert choice.delta.role is None
            assert choice.delta.content is None
            continue

        if choice.delta.role is not None:
            role_chunks += 1
            assert index == 0
            assert choice.delta.role == "assistant"
            assert choice.delta.content is None
            continue

        assert isinstance(choice.delta.content, str)
        content_chunks += 1

    assert role_chunks == 1
    assert content_chunks >= 1
    assert terminal_chunks == 1
    assert usage_chunks == 1


if __name__ == "__main__":
    main()
