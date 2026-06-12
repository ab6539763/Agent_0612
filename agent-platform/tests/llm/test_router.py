"""ProviderRouter 测试。"""

from __future__ import annotations

import pytest
from src.core.exceptions import ValidationError
from src.llm.router import ProviderRouter

from tests.llm.fakes import ScriptedProvider, make_request, make_result


class TestProviderRouter:
    async def test_routes_by_prefix(self) -> None:
        openai_provider = ScriptedProvider([make_result("from-openai")])
        vllm_provider = ScriptedProvider([make_result("from-vllm")])
        router = ProviderRouter({"openai": openai_provider, "vllm": vllm_provider})

        result = await router.complete(make_request("vllm:qwen2.5-7b"))

        assert result.message.content == "from-vllm"
        assert vllm_provider.calls == 1
        assert openai_provider.calls == 0

    async def test_stream_routes(self) -> None:
        provider = ScriptedProvider([make_result("streamed")])
        router = ProviderRouter({"openai": provider})

        chunks = [c async for c in router.stream(make_request("openai:gpt-4o"))]
        assert chunks[0].content_delta == "streamed"

    async def test_unknown_prefix_rejected(self) -> None:
        router = ProviderRouter({"openai": ScriptedProvider([make_result()])})
        with pytest.raises(ValidationError, match="no provider registered"):
            await router.complete(make_request("mystery:model-x"))

    async def test_unqualified_model_rejected(self) -> None:
        router = ProviderRouter({"openai": ScriptedProvider([make_result()])})
        with pytest.raises(ValidationError, match="qualified"):
            await router.complete(make_request("gpt-4o"))

    def test_register_and_prefixes(self) -> None:
        router = ProviderRouter({})
        router.register("openai", ScriptedProvider([make_result()]))
        assert router.prefixes == frozenset({"openai"})
