"""Wrapper for flattening observations of an environment."""
from typing import Union

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector.utils import batch_space


def FlattenObservation(
    env: Union[gym.Env, gym.vector.VectorEnv]
) -> Union[gym.ObservationWrapper, gym.vector.VectorObservationWrapper]:
    """Return appropriate ClipAction wrapper."""
    if isinstance(env, gym.vector.VectorEnv):
        return _VectorFlattenObservation(env)
    else:
        return _FlattenObservation(env)


class _FlattenObservation(gym.ObservationWrapper):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs, info = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        return spaces.flatten(self.env.observation_space, observation)


class _VectorFlattenObservation(gym.vector.VectorObservationWrapper):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.make('CarRacing-v1', 8)
        >>> env.observation_space.shape
        (8, 96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (8, 27648)
        >>> obs, info = env.reset()
        >>> obs.shape
        (8, 27648)
    """

    def __init__(self, env: gym.vector.VectorEnv):
        """Flattens the observations of an environment.

        Args:
            env: The vector environment to apply the wrapper
        """
        super().__init__(env)
        self.single_observation_space = spaces.flatten_space(
            env.single_observation_space
        )
        self.observation_space = batch_space(
            self.single_observation_space, env.num_envs
        )

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        # TODO: Flatten all but vector dimension
        return spaces.flatten(self.env.observation_space, observation)
