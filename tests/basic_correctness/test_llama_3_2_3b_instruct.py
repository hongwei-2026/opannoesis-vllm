# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for meta-llama/Llama-3.2-3B-Instruct adaptation (Issue #15).

Verifies the model can be loaded by vLLM and complete at least one inference.
"""

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def test_llama_3_2_3b_instruct_basic_generation():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        max_model_len=1024,
        enforce_eager=True,
    )
    prompts = [
        "The capital of France is",
        "Write a one-sentence introduction of yourself.",
    ]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=32))
    assert len(outputs) == 2
    for out in outputs:
        text = out.outputs[0].text
        assert text is not None and len(text.strip()) > 0, (
            f"Empty generation for prompt: {out.prompt!r}"
        )
