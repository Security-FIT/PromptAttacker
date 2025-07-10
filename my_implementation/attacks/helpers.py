# TENTO FILE JE MUUUUUUUUUUJ

import os
import json
import yaml
from typing import List, Dict
import abc
from typing import Dict, List, Union, Any
from dataclasses import dataclass, field



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