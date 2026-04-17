
import math
import operator
import collections
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Mapping, Set, Tuple, TypeVar


T = TypeVar("T")


def fact_div(a: int, b: int) -> float:
    """return a! / b!"""
    return product(range(b + 1, a + 1)) if a >= b else 1.0 / fact_div(b, a)


def choose(n: int, k: int) -> float:
    """return n choose k

    resilient (though not immune) to integer overflow"""
    if n == 1:
        # optimize by far most-common case
        return 1

    return fact_div(n, max(k, n - k)) / math.factorial(min(k, n - k))


def peek(iterable: Iterable[T]) -> T:
    """return an arbitrary item from a collection; no ordering is guaranteed

    useful for extracting singletons, or when you're managing iteration
    yourself"""
    return next(iter(iterable))


def product(n: Iterable[float | int]) -> float:
    """return the product of a set of numbers

    n -- an iterable of numbers"""
    return reduce(operator.mul, n, 1)


def listify(x: Any) -> List[Any]:
    """convert object to a list; if not an iterable, wrap as a list of length 1"""
    return list(x) if hasattr(x, "__iter__") else [x]


def graph_traverse(graph: Mapping[T, Iterable[T]], node: T) -> Set[T]:
    """graph traversal algorithm -- given a graph and a node, return the set
    of nodes that can be reached from 'node', including 'node' itself"""
    visited: Set[T] = set()

    def _graph_traverse(n: T) -> None:
        visited.add(n)
        for neighbor in graph[n]:
            if neighbor not in visited:
                _graph_traverse(neighbor)

    _graph_traverse(node)
    return visited


def _default_emit(rec: Any) -> List[Tuple[Any]]:
    return [(rec,)]


def _default_reduce(values: List[Any]) -> Any:
    return values


def map_reduce(
    data: Iterable[Any],
    emitfunc: Callable[[Any], Iterable[Tuple[Any, Any] | Tuple[Any]]] = _default_emit,
    reducefunc: Callable[[List[Any]], Any] = _default_reduce,
) -> Dict[Any, Any]:
    """perform a "map-reduce" on the data

    emitfunc(datum): return an iterable of key-value pairings as (key, value). alternatively, may
        simply emit (key,) (useful for reducefunc=len)
    reducefunc(values): applied to each list of values with the same key; defaults to just
        returning the list
    data: iterable of data to operate on
    """
    mapped: Dict[Any, List[Any]] = collections.defaultdict(list)
    for rec in data:
        for emission in emitfunc(rec):
            if len(emission) == 2:
                k, v = emission
            else:
                k, v = emission[0], None
            mapped[k].append(v)
    return dict((k, reducefunc(v)) for k, v in mapped.items())


class ImmutableMixin(object):
    """mixin for immutable, hashable objects"""

    def _canonical(self) -> Tuple[Any, ...]:
        """return the 'core' data of this object in a hashable format, usually a tuple"""
        assert False, "must override"

    def _cached_canonical(self) -> Tuple[Any, ...]:
        try:
            return self._immutable_canonical
        except AttributeError:
            self._immutable_canonical = self._canonical()
            return self._immutable_canonical

    def __eq__(self, o: Any) -> bool:
        if self is o:
            return True
        return type(self) == type(o) and self._cached_canonical() == o._cached_canonical()

    def __ne__(self, o: Any) -> bool:
        return not (self == o)

    def __hash__(self) -> int:
        try:
            return self._immutable_hash
        except AttributeError:
            self._immutable_hash = hash(self._cached_canonical())
            return self._immutable_hash
