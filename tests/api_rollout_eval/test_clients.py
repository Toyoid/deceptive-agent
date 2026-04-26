import asyncio
import types

from agent_system.api_rollout_eval.clients import OpenAICompatibleChatClient


class _FakeCompletions:
    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    async def create(self, **kwargs):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("temporary")
        message = types.SimpleNamespace(content=f"answer-{self.calls}")
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeAsyncClient:
    def __init__(self, completions):
        self.chat = types.SimpleNamespace(completions=completions)
        self.closed = False

    async def close(self):
        self.closed = True


def test_client_uses_dummy_key_for_local_endpoint_and_generates():
    async def _run():
        completions = _FakeCompletions()
        fake_client = _FakeAsyncClient(completions)
        client = OpenAICompatibleChatClient(
            model="local-model",
            api_base="http://127.0.0.1:8000/v1",
            api_key=None,
            api_key_env="MISSING_API_KEY_FOR_TEST",
            async_client=fake_client,
        )

        assert client.api_key == "dummy"
        responses = await client.generate_batch([[{"role": "user", "content": "hi"}]])

        assert responses[0].text == "answer-1"
        assert responses[0].finish_reason == "stop"
        assert responses[0].total_tokens == 7
        await client.close()
        assert fake_client.closed

    asyncio.run(_run())


def test_client_retries_and_returns_ordered_responses():
    async def _run():
        completions = _FakeCompletions(fail_first=True)
        client = OpenAICompatibleChatClient(
            model="local-model",
            api_base="http://localhost:8000/v1",
            max_retries=2,
            max_concurrent=1,
            retry_delay=0.0,
            async_client=_FakeAsyncClient(completions),
        )

        responses = await client.generate_batch([
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ])

        assert [response.error for response in responses] == [None, None]
        assert [response.text for response in responses] == ["answer-2", "answer-3"]

    asyncio.run(_run())
