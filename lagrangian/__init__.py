"""Lagrangian neural networks for mechanical systems.

Provides:
- LagNet: Pendulum (1-DoF)
- CartPoleLagNet: CartPole (2-DoF) with structured mass matrix
- AcrobotLagNet: Acrobot (2-DoF underactuated) with structured mass matrix
- DeLaN (future): General n-DoF without system-specific structure
"""
from lagrangian._pendulum import LagNet
from lagrangian._cartpole import CartPoleLagNet
from lagrangian._acrobot import AcrobotLagNet

__all__ = ['LagNet', 'CartPoleLagNet', 'AcrobotLagNet']
