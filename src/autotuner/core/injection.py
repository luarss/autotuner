"""Dependency injection container for AutoTuner components."""

from collections.abc import Callable
from typing import Any, TypeVar

from .interfaces import Algorithm, Evaluator, Runner

T = TypeVar("T")


class DIContainer:
    """Simple dependency injection container."""

    def __init__(self):
        self._bindings: dict[type, Any] = {}
        self._singletons: dict[type, Any] = {}

    def bind(self, interface: type[T], implementation: type[T]) -> None:
        """Bind an interface to a concrete implementation.

        Args:
            interface: Abstract interface class
            implementation: Concrete implementation class
        """
        self._bindings[interface] = implementation

    def bind_singleton(self, interface: type[T], implementation: type[T]) -> None:
        """Bind an interface to a singleton implementation.

        Args:
            interface: Abstract interface class
            implementation: Concrete implementation class
        """
        self._bindings[interface] = implementation
        self._singletons[interface] = None

    def bind_instance(self, interface: type[T], instance: T) -> None:
        """Bind an interface to a specific instance.

        Args:
            interface: Abstract interface class
            instance: Concrete instance
        """
        self._bindings[interface] = instance
        self._singletons[interface] = instance

    def get(self, interface: type[T], **kwargs) -> T:
        """Get an instance of the specified interface.

        Args:
            interface: Abstract interface class
            **kwargs: Additional parameters to pass to constructor

        Returns:
            Instance of the concrete implementation

        Raises:
            ValueError: If interface is not bound
        """
        if interface not in self._bindings:
            raise ValueError(f"Interface {interface} is not bound")

        implementation = self._bindings[interface]

        # If it's already an instance, return it
        if not isinstance(implementation, type):
            return implementation

        # Check if it's a singleton
        if interface in self._singletons:
            if self._singletons[interface] is None:
                self._singletons[interface] = implementation(**kwargs)
            return self._singletons[interface]

        # Create new instance
        return implementation(**kwargs)

    def is_bound(self, interface: type[T]) -> bool:
        """Check if an interface is bound.

        Args:
            interface: Abstract interface class

        Returns:
            True if interface is bound, False otherwise
        """
        return interface in self._bindings

    def unbind(self, interface: type[T]) -> None:
        """Remove binding for an interface.

        Args:
            interface: Abstract interface class
        """
        if interface in self._bindings:
            del self._bindings[interface]
        if interface in self._singletons:
            del self._singletons[interface]


# Global container instance
container = DIContainer()


def configure_container() -> DIContainer:
    """Configure the dependency injection container with default bindings.

    Returns:
        Configured DI container
    """
    # Note: Actual implementations would be bound here
    # This is a placeholder for when concrete implementations are created
    return container


def inject(interface: type[T]) -> Callable[[Callable], Callable]:
    """Decorator to inject dependencies into functions.

    Args:
        interface: Interface to inject

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if interface.__name__.lower() not in kwargs:
                kwargs[interface.__name__.lower()] = container.get(interface)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_algorithm() -> Algorithm:
    """Get the configured algorithm instance.

    Returns:
        Algorithm instance
    """
    return container.get(Algorithm)


def get_evaluator() -> Evaluator:
    """Get the configured evaluator instance.

    Returns:
        Evaluator instance
    """
    return container.get(Evaluator)


def get_runner() -> Runner:
    """Get the configured runner instance.

    Returns:
        Runner instance
    """
    return container.get(Runner)
