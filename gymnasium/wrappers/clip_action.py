"""Wrapper for clipping actions within a valid bound."""
from typing import Union

import numpy as np

import gymnasium as gym
from gymnasium import ActionWrapper
from gymnasium.spaces import Box
from gymnasium.vector import VectorActionWrapper


def ClipAction(
    env: Union[gym.Env, gym.vector.VectorEnv]
) -> Union[ActionWrapper, VectorActionWrapper]:
    """Return appropriate ClipAction wrapper."""
    if isinstance(env, gym.vector.VectorEnv):
        return _VectorClipAction(env)
    else:
        return _ClipAction(env)


class _ClipAction(ActionWrapper):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('Bipedal-Walker-v3')
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env.step(np.array([5.0, 2.0, -10.0, 0.0]))
        # Executes the action np.array([1.0, 1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)
        super().__init__(env)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(action, self.action_space.low, self.action_space.high)


class _VectorClipAction(VectorActionWrapper):
    """Clip the continuous actions within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.make('Bipedal-Walker-v3', 8)
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (8, 4), float32)
        >>> env.step(np.array([[5.0, 2.0, -10.0, 0.0]] * 8))
        # Executes the action np.array([1.0, 1.0, -1.0, 0]) in each base environment
    """

    def __init__(self, env: gym.vector.VectorEnv):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)
        assert isinstance(env.single_action_space, Box)
        super().__init__(env)

    def actions(self, actions):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(actions, self.action_space.low, self.action_space.high)
