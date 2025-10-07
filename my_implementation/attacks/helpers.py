from __future__ import annotations

# TENTO FILE JE MUUUUUUUUUUJ

import os
import json
import yaml
import argparse
from typing import List, Dict
import abc
from typing import Dict, List, Union, Any
from dataclasses import dataclass, field
from vllm import LLM as VLLMClient, SamplingParams

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r', encoding='utf-8') as f:
        if ext in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file: {path}")
        

def is_assistant_turn(messages: List[Dict[str, str]]) -> bool:
    """
    Check if it's the assistant's turn based on the last message.

    :param messages: List of message dictionaries containing "role" and "content".
    :return: True if the last message is from the assistant, False otherwise.
    """
    return messages and len(messages) > 0 and messages[-1]["role"] == "assistant"


def is_user_turn(messages: List[Dict[str, str]]) -> bool:
    """
    Check if it's the user's turn based on the last message.

    :param messages: List of message dictionaries containing "role" and "content".
    :return: True if the last message is from the user, False otherwise.
    """
    return messages and len(messages) > 0 and messages[-1]["role"] == "user"


@dataclass
class BaseAttackerConfig(abc.ABC):
    """
    Configuration for the Base Attacker.

    :param attacker_cls: Class of the attacker.  
    :param attacker_name: Name of the attacker.  
    """
    attacker_cls: str = field(default=None)
    attacker_name: str = field(default=None)


class BaseAttacker(abc.ABC):
    """
    Abstract Base Class for Attacker.

    :param config: Configuration for the attacker. 
    """

    def __init__(
            self,
            config: BaseAttackerConfig
    ):
        self._CLS = config.attacker_cls
        self._NAME = config.attacker_name

    @abc.abstractmethod
    def attack(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> List[Dict[str, str]]:
        """
        Abstract method to execute an attack on a sequence of messages.

        :param messages: List of input messages.  
        :param kwargs: Additional parameters for the attack.  
        :return: Modified list of messages after the attack. 
        """
        if messages is None:
            messages = []
        else:
            assert messages[-1]["role"] != "user"

        pass


@dataclass
class BaseJudgeConfig(abc.ABC):
    """
    Base configuration for the Judge class.

    :param judge_cls: Class of the judge, default is None.
    :param judge_name: Name of the judge, default is None.
    """
    judge_cls: str = field(default=None)
    judge_name: str = field(default=None)


class BaseJudge(abc.ABC):
    """
    Base class for implementing a judge to evaluate the safety of a given response.

    :param config: Configuration for the judge.
    """

    def __init__(
            self,
            config: BaseJudgeConfig,
    ):
        self._CLS = config.judge_cls
        self._NAME = config.judge_name

    @abc.abstractmethod
    def judge(
            self,
            messages: List[Dict[str, str]] = None,
            request: str = None
    ) -> int:
        """
        Abstract method to evaluate the safety of a given request and messages.

        :param messages: List of messages to evaluate.
        :param request: The user's request.
        :return: An integer representing the evaluation result (0: Unsafe, 1: Safe).
        """
        pass




class BaseLLM:  # pylint: disable=too-few-public-methods
    """Very small wrapper around *vLLM* exposing only :py:meth:`response`."""

    def __init__(
        self,
        model_path: str,
        *,
        temperature: float = 0.8,
        max_tokens: int = 512,
        **client_kwargs: Any,
    ) -> None:
        """Create the wrapper.

        Parameters
        ----------
        model_path
            Filesystem path *or* HuggingFace model ID.
        temperature
            Sampling temperature.
        max_tokens
            Maximum number of *new* tokens to generate.
        **client_kwargs
            Any extra keyword arguments are forwarded directly to
            :class:`vllm.LLM` – e.g. ``tensor_parallel_size`` or
            ``gpu_memory_utilization``.
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Lazily‑constructed vLLM engine.  We create it here so that the same
        # process can reuse the weights for multiple calls.
        self._client = VLLMClient(
            model=model_path,
            trust_remote_code=True,
            **client_kwargs,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def response(self, messages: List[Dict[str, str]]) -> str:  # noqa: D401 – simple method
        """Generate a single response string.

        The *last* element in *messages* is treated as the user prompt.  All
        preceding items are ignored in this minimalist variant – if you want
        proper chat formatting, prepend it to the last ``content`` yourself.
        """
        if not messages:
            raise ValueError("`messages` must contain at least one item.")

        user_prompt: str = messages[-1]["content"]

        # Call vLLM.  The API returns a list of request objects; each contains a
        # list of candidate completions.  We take the top‑1 text.
        outputs = self._client.generate(
            [{"prompt": user_prompt}],
            sampling_params=SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )

        return outputs[0].outputs[0].text.strip()


@dataclass
class BaseLLMConfig(abc.ABC):
    """
    Base configuration for LLM.

    :param llm_type: Type of the LLM.
    :param model_name: Name of the model.
    """

    llm_type: str = field(default=None)
    model_name: str = field(default=None)

@dataclass
class LLMGenerateConfig:
    """
    Configuration for LLM generation.

    :param max_n_tokens: Maximum number of tokens to generate.
    :param temperature: Temperature for sampling randomness.
    :param logprobs: Whether to return log probabilities.
    :param seed: Seed for reproducibility.
    :param stream: Whether to use streaming generation.
    """

    max_n_tokens: int = field(default=None)
    temperature: float = field(default=None)
    logprobs: bool = field(default=False)
    seed: int = field(default=None)
    stream: bool = field(default=False)  # Default to non-streaming behavior