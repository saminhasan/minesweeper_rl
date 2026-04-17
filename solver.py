from __future__ import annotations
import heapq
import operator
import itertools
import collections
from solver_util import *
from functools import reduce
from typing import Any, Set, Dict, List, Self, Tuple, Union, Iterator, Optional, Callable, Mapping, Iterable


set_ = frozenset


class InconsistencyError(Exception):
    """raise when a game state is logically inconsistent."""

    pass


"""represents the board geometry for traditional minesweeper, where the board
has fixed dimensions and fixed total # of mines.

total_cells -- total # of cells on board; all cells contained in rules + all
    'uncharted' cells
total_mines: total # of mines contained within all cells
"""
MineCount = collections.namedtuple("MineCount", ["total_cells", "total_mines"])


class Rule(ImmutableMixin):
    """basic representation of an axiom from a minesweeper game: N mines
    contained within a set of M cells.

    only used during the very early stages of the algorithm; quickly converted
    to 'Rule_'

    num_mines -- # of mines
    cells -- list of cells; each 'cell' is a unique, identifying tag that
        represents that cell (string, int, any hashable)
    """

    def __init__(self, num_mines: int, cells: List[Any]) -> None:
        self.num_mines = num_mines
        self.cells = set_(cells)

    def condensed(self, rule_supercells_map: Dict[Self, frozenset]) -> Rule_:
        """condense supercells and convert to a 'Rule_'

        rule_supercells_map -- pre-computed supercell mapping
        """
        return Rule_(
            self.num_mines,
            rule_supercells_map.get(self, set_()),  # default to handle degenerate rules
            len(self.cells),
        )

    def _canonical(self) -> Tuple[int, frozenset]:
        return (self.num_mines, self.cells)

    def __repr__(self):
        return "Rule(num_mines=%d, cells=%s)" % (
            self.num_mines,
            sorted(list(self.cells)),
        )


class Permutation(ImmutableMixin):
    """a single permutation of N mines among a set of (super)cells"""

    def __init__(self, mapping: Union[Dict[frozenset, int], Iterator, Set[Tuple[frozenset, int]]]) -> None:
        """mapping -- a mapping: supercell -> # of mines therein

        cell set is determined implicitly from mapping, so all cells in set
        must have an entry, even if they have 0 mines"""
        self.mapping = dict(mapping)
        self._cells = set_(self.mapping.keys())
        self._k = sum(self.mapping.values())
        self._multiplicity: Optional[float] = None

    def subset(self, subcells: Union[Set[frozenset], frozenset]) -> Permutation:
        """return a sub-permutation containing only the cells in 'subcells'"""
        return Permutation((cell, self.mapping[cell]) for cell in subcells)

    def compatible(self, permu: Self) -> bool:
        """return whether this permutation is consistent with 'permu', meaning
        the cells they have in common have matching numbers of mines assigned"""
        if len(self.mapping) <= len(permu.mapping):
            smaller, larger = self.mapping, permu.mapping
        else:
            smaller, larger = permu.mapping, self.mapping

        for cell_, mines in smaller.items():
            other = larger.get(cell_)
            if other is not None and other != mines:
                return False
        return True

    def combine(self, permu: Self) -> Permutation:
        """return a new permutation by combining this permutation with
        'permu'
        the permutations must be compatible!"""
        return Permutation({**self.mapping, **permu.mapping})

    def k(self) -> int:
        """return total # mines in this permutation"""
        return self._k

    def cells(self) -> frozenset:
        """return set of cells in this permutation"""
        return self._cells

    def multiplicity(self) -> float:
        """count the # of permutations this permutation would correspond to if
        each supercell were broken up into singleton cells.

        e.g., N mines in a supercell of M cells has (M choose N) actual
        configurations
        """
        if self._multiplicity is None:
            self._multiplicity = product(choose(len(cell_), k) for cell_, k in self.mapping.items())
        return self._multiplicity

    def _canonical(self) -> frozenset:
        return set_(self.mapping.items())

    def __repr__(self):
        cell_counts = sorted([(sorted(list(cell)), count) for cell, count in self.mapping.items()])
        cell_frags = ["%s:%d" % (",".join(str(c) for c in cell), count) for cell, count in cell_counts]
        return "{%s}" % " ".join(cell_frags)


class UnchartedCell(ImmutableMixin):
    """a meta-cell object that represents all the 'other' cells on the board
    that aren't explicitly mentioned in a rule. see expand_cells()"""

    def __init__(self, size: int = 1) -> None:
        """
        size -- # of 'other' cells
        """
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator:
        """only appear once in the solution, regardless of size. however,
        don't appear at all if in fact there are no 'other' cells"""
        if self.size > 0:
            yield None

    def _canonical(self) -> Tuple[int]:
        return (self.size,)


class Rule_(ImmutableMixin):
    """analogue of 'Rule', but containing supercells (sets of 'ordinary' cells
    that only ever appear together).

    this is the common rule form used throughout most of the algorithm

    num_mines -- total # of mines
    num_cells -- total # of base cells
    cells_ -- set of supercells; each supercell a set of base cells
    """

    def __init__(
        self,
        num_mines: int,
        cells_: frozenset,
        num_cells: Optional[int] = None,
    ) -> None:
        self.num_mines = num_mines
        self.cells_ = cells_
        self.num_cells = num_cells if num_cells is not None else sum(len(cell_) for cell_ in cells_)

        if self.num_mines < 0 or self.num_mines > self.num_cells:
            raise InconsistencyError("rule with negative mines / more mines than cells")

    def decompose(self) -> Iterator[Rule_]:
        """if rule is completely full or empty of mines, split into sub-rules
        for each supercell"""
        if self.num_mines == 0 or self.num_mines == self.num_cells:
            for cell_ in self.cells_:
                size = len(cell_)
                yield Rule_(size if self.num_mines > 0 else 0, set_([cell_]), size)
            # degenerate rules (no cells) disappear here
        else:
            yield self

    def subtract(self, subrule: Rule_) -> Rule_:
        """if another rule is a sub-rule of this one, return a new rule
        covering only the difference"""
        return Rule_(
            self.num_mines - subrule.num_mines,
            self.cells_ - subrule.cells_,
            self.num_cells - subrule.num_cells,
        )

    def permute(self) -> Iterator[Permutation]:
        """generate all possible mine permutations of this rule"""
        for p in permute(self.num_mines, list(self.cells_)):
            yield p

    def is_subrule_of(self, parent: Self) -> bool:
        """return if this rule is a sub-rule of 'parent'

        'sub-rule' means this rule's cells are a subset of the parent rules'
        cells. equivalent rules are subrules of each other.
        """
        return self.cells_.issubset(parent.cells_)

    def is_trivial(self) -> bool:
        """return whether this rule is trivial, i.e., has only one permutation"""
        return len(self.cells_) == 1

    def tally(self) -> FrontTally:
        """build a FrontTally from this *trivial* rule only"""
        return FrontTally.from_rule(self)

    def _canonical(self) -> Tuple[int, frozenset]:
        return (self.num_mines, self.cells_)

    def __repr__(self):
        return "Rule_(num_mines=%d, num_cells=%d, cells_=%s)" % (
            self.num_mines,
            self.num_cells,
            sorted([sorted(list(cell_)) for cell_ in self.cells_]),
        )

    #####################################################################
    @staticmethod
    def mk(num_mines, cells_):
        """helper method for creation

        num_mines -- total # of mines
        cells_ -- list of cells and supercells, where a supercell is a list of
            ordinary cells, e.g., ['A', ['B', 'C'], 'D']
        """
        cells_ = [listify(cell_) for cell_ in cells_]
        return Rule_(num_mines, set_(set_(cell_) for cell_ in cells_))


class PermutedRuleset(object):
    """a set of rules and the available permutations for each, eliminating
    permutations which are mutually-inconsistent across the ruleset"""

    def __init__(
        self,
        rules: set[Rule_],
        permu_map: Optional[Dict[Rule_, PermutationSet]] = None,
    ) -> None:
        """
        rules -- ruleset
        permu_map -- if creating a subset of another PermutedRuleset, will be
            the permu_map of the parent; for a new PermutedRuleset, will be
            computed automatically
        """
        self.rules = rules
        self.cell_rules_map = CellRulesMap(rules)
        self.cells_ = self.cell_rules_map.cells_()

        def rule_permuset(r: Rule_) -> PermutationSet:
            return PermutationSet.from_rule(r) if permu_map is None else permu_map[r]

        # a mapping: rule -> PermutationSet for that rule
        self.permu_map = dict((rule, rule_permuset(rule)) for rule in rules)

    def cross_eliminate(self) -> None:
        """determine what permutations are possible for each rule, taking
        into account the constraints of all overlapping rules. eliminate
        impossible permutations"""

        interferences = self.cell_rules_map.interference_edges()

        # we can't simply iterate through 'interferences', as eliminating a
        # permutation in a rule may in turn invalidate permutations in other
        # overlapping rules that have already been processed, thus causing a
        # cascade effect
        while interferences:
            r, r_ov = interferences.pop()
            changed = False
            ov_permuset = self.permu_map[r_ov]
            for permu in tuple(self.permu_map[r]):  # copy iterable so we can modify original
                if not ov_permuset.has_compatible(permu):
                    # this permutation has no compatible permutation in the overlapping
                    # rule. thus, it can never occur
                    self.permu_map[r].remove(permu)
                    changed = True

            if self.permu_map[r].empty():
                # no possible configurations for this rule remain
                raise InconsistencyError("rule is constrained such that it has no valid mine permutations")
            elif changed:
                # other rules overlapping with this one must be recalculated
                for r_other in self.cell_rules_map.overlapping_rules(r):
                    interferences.add((r_other, r))

    def rereduce(self) -> None:
        """after computing the possible permutations of the rules, analyze and
        decompose rules into sub-rules, if possible. this can eliminate
        dependencies among the initial set of rules, and thus potentially
        split what would have been one rule-front into several.

        this is analagous to the previous 'reduce_rules' step, but with more
        advanced logical analysis -- exploiting information gleaned from the
        permutation phase
        """

        """
        postulates that i'm pretty certain about, but can't quite prove
        *) among all cartesian decompositions from all rules, none will be reduceable with another
           (decomp'ed rules may have duplicates, though)
        *) cartesian decomposition will have effectively re-reduced all rules in the set, even non-
           decomp'ed rules; there will be no possible reductions between a decomp'ed rule and an
           original rule
        *) re-permuting amongst the de-comped ruleset will produce the same permutation sets
        """

        superseded_rules = set()
        decompositions: Dict = {}
        for rule, permu_set in self.permu_map.items():
            decomp = permu_set.decompose()
            if len(decomp) > 1:
                superseded_rules.add(rule)
                # collapse duplicate decompositions by keying by cell set
                decompositions.update((dc.cells_, dc) for dc in decomp)

        for rule in superseded_rules:
            self.remove_rule(rule)
        for permu_set in list(decompositions.values()):
            self.add_permu_set(permu_set)

    def remove_rule(self, rule: Rule_) -> None:
        self.rules.remove(rule)
        self.cell_rules_map.remove_rule(rule)
        del self.permu_map[rule]

    def add_permu_set(self, permu_set: PermutationSet) -> None:
        """add a 'decomposed' rule to the ruleset"""
        rule = permu_set.to_rule()
        self.rules.add(rule)
        self.cell_rules_map.add_rule(rule)
        self.permu_map[rule] = permu_set

    def filter(self, rule_subset: Set[Rule_]) -> PermutedRuleset:
        """return a PermutedRuleset built from this one containing only a
        subset of rules"""
        return PermutedRuleset(rule_subset, self.permu_map)

    def split_fronts(self) -> Set[PermutedRuleset]:
        """split the ruleset into combinatorially-independent 'fronts'"""
        return set(self.filter(rule_subset) for rule_subset in self.cell_rules_map.partition())

    def is_trivial(self) -> bool:
        """return whether this ruleset is trivial, i.e., contains only one rule"""
        return len(self.rules) == 1

    def trivial_rule(self) -> Rule_:
        """return the singleton rule of this *trivial* ruleset"""
        assert self.is_trivial()
        singleton = peek(self.rules)

        # postulate: any singleton rule must also be trivial
        assert singleton.is_trivial()

        return singleton

    def enumerate(self) -> Iterator[Permutation]:
        """enumerate all possible mine configurations for this ruleset"""
        for mineconfig in EnumerationState(self).enumerate():
            yield mineconfig

    def __repr__(self):
        import pprint

        return "PermutedRuleset(\n %s)" % pprint.pformat(self.permu_map)


class FrontTally(object):
    """tabulation of per-cell mine frequencies"""

    def __init__(self, data: Any = None) -> None:
        # mapping: # of mines in configuration -> sub-tally of configurations with that # of mines
        self.subtallies = collections.defaultdict(FrontSubtally) if data is None else data
        self.total = None

    def tally(self, front: PermutedRuleset) -> None:
        """tally all possible configurations for a front (ruleset)

        note that the tallies for different total # of mines must be
        maintained separately, as these will be given different statistical
        weights later on
        """

        for config in front.enumerate():
            self.subtallies[config.k()].add(config)

        if not self.subtallies:
            # front has no possible configurations
            raise InconsistencyError("mine front has no possible configurations")

        self.finalize()

    def finalize(self) -> None:
        """finalize all sub-tallies (convert running totals to
        probabilities/expected values)"""
        for subtally in list(self.subtallies.values()):
            subtally.finalize()

    def min_mines(self) -> int:
        """minimum # of mines found among all configurations"""
        return min(self.subtallies)

    def max_mines(self) -> int:
        """maximum # of mines found among all configurations"""
        return max(self.subtallies)

    def is_static(self) -> bool:
        """whether all configurations have the same # of mines (simplifies
        statistical weighting later)"""
        return len(self.subtallies) == 1

    def __iter__(self) -> Iterator:
        return iter(self.subtallies.items())

    def normalize(self) -> None:
        """normalize sub-tally totals into relative weights such that
        sub-totals remain proportional to each other, and the grand total
        across all sub-tallies is 1."""
        total = sum(subtally.total for subtally in list(self.subtallies.values()))
        for subtally in list(self.subtallies.values()):
            subtally.total /= float(total)
            subtally.normalized = True

    def collapse(
        self,
    ) -> Iterator[Tuple[Any, float]]:
        """calculate the per-cell expected mine values, summed/weighted across
        all sub-tallies"""
        self.normalize()
        collapsed: Dict[Any, float] = {}
        for subtally in self.subtallies.values():
            for cell_, value in subtally.collapse():
                collapsed[cell_] = collapsed.get(cell_, 0.0) + value
        for entry in collapsed.items():
            yield entry

    def scale_weights(self, scalefunc: Callable[[int], float]) -> None:
        """scale each sub-tally's weight/total according to 'scalefunc'

        scalefunc -- function: num_mines -> factor by which to scale the sub-tally for 'num_mines'
        """
        for num_mines, subtally in self:
            subtally.total *= scalefunc(num_mines)

    def update_weights(self, weights: dict[int, float] | dict[int, int]) -> None:
        """update each sub-tally's weight/total

        weights -- mapping: num_mines -> new weight of the sub-tally for 'num_mines'
        """
        for num_mines, subtally in self:
            subtally.total = weights.get(num_mines, 0.0)

    @staticmethod
    def from_rule(rule: Rule_) -> FrontTally:
        """tally a trivial rule"""
        assert rule.is_trivial()
        return FrontTally(
            {
                rule.num_mines: FrontSubtally.mk(
                    choose(rule.num_cells, rule.num_mines),
                    {peek(rule.cells_): rule.num_mines},
                )
            }
        )

    @staticmethod
    def for_other(num_uncharted_cells: int, mine_totals: dict[int, float] | dict[int, int]) -> FrontTally:
        """create a meta-tally representing the mine distribution of all
        'other' cells

        num_uncharted_cells -- # of 'other' cells
        mine_totals -- a mapping suitable for update_weights(): # mines in 'other' region -> relative likelihood
        """

        metacell = UnchartedCell(num_uncharted_cells)
        return FrontTally(dict((num_mines, FrontSubtally.mk(k, {metacell: num_mines})) for num_mines, k in mine_totals.items()))

    def __repr__(self):
        return str(dict(self.subtallies))


class Reduceable(ImmutableMixin):
    """during the logical deduction phase, if all rules are nodes in a graph,
    this represents a directed edge in that graph indicating 'superrule' can
    be reduced by 'subrule'"""

    def __init__(self, superrule: Rule_, subrule: Rule_) -> None:
        self.superrule = superrule
        self.subrule = subrule

    def __lt__(self, other: Self) -> bool:
        # Implement a comparison based on the appropriate attributes of Reduceable
        # For example, if Reduceable has a 'metric' method that returns a value that can be compared:
        return self.metric() < other.metric()

    def metric(self) -> Tuple[int, int, float]:
        """calculate the attractiveness of this reduction

        favor reductions that involve bigger rules, and amongst same-sized
        rules, those that yield # mines towards the extremes -- such rules
        have fewer permutations
        """

        num_reduced_cells = self.superrule.num_cells - self.subrule.num_cells
        num_reduced_mines = self.superrule.num_mines - self.subrule.num_mines
        return (
            self.superrule.num_cells,
            self.subrule.num_cells,
            abs(num_reduced_mines - 0.5 * num_reduced_cells),
        )

    def reduce(self) -> Rule_:
        """perform the reduction"""
        return self.superrule.subtract(self.subrule)

    def contains(self, rule: Rule_) -> bool:
        return rule in (self.superrule, self.subrule)

    def contained_within(self, rules: Set[Rule_]) -> bool:
        return all(r in rules for r in (self.superrule, self.subrule))

    def _canonical(self) -> Tuple[Rule_, Rule_]:
        return (self.superrule, self.subrule)

    def __repr__(self):
        return "Reduceable(superrule=%s, subrule=%s)" % (self.superrule, self.subrule)


class CellRulesMap(object):
    """a utility class mapping cells to the rules they appear in"""

    def __init__(self, rules: Optional[Union[List[Rule_], Set[Rule_], frozenset]] = None) -> None:
        # a mapping: cell -> list of rules cell appears in
        self.map: Dict[frozenset, Set[Rule_]] = collections.defaultdict(set)
        self.rules: Set = set()
        self.add_rules(rules or [])

    def add_rules(self, rules: Union[List, Set[Rule_], frozenset]) -> None:
        for rule in rules:
            self.add_rule(rule)

    def add_rule(self, rule: Rule_) -> None:
        self.rules.add(rule)
        for cell_ in rule.cells_:
            self.map[cell_].add(rule)

    def remove_rule(self, rule: Rule_) -> None:
        self.rules.remove(rule)
        for cell_ in rule.cells_:
            cell_rules = self.map[cell_]
            cell_rules.remove(rule)
            if not cell_rules:
                del self.map[cell_]

    def overlapping_rules(self, rule: Rule_) -> Set[Rule_]:
        """Return a set of rules that overlap with 'rule', i.e., have at least one cell in common."""
        overlapping: Set[Rule_] = set()
        for cell_ in rule.cells_:
            cell_rules = self.map.get(cell_)
            if cell_rules:
                overlapping.update(cell_rules)

        overlapping.discard(rule)
        return overlapping

    def interference_edges(self) -> Set[Tuple[Rule_, Rule_]]:
        """return pairs of all rules that overlap each other; each pair is
        represented twice ((a, b) and (b, a)) to support processing of
        relationships that are not symmetric"""

        def _interference_edges() -> Iterator[Tuple[Rule_, Rule_]]:
            for rule in self.rules:
                for rule_ov in self.overlapping_rules(rule):
                    yield (rule, rule_ov)

        return set(_interference_edges())

    def partition(self) -> Set:
        """partition the ruleset into disjoint sub-rulesets of related rules.

        that is, all rules in a sub-ruleset are related to each other in some
        way through some number of overlaps, and no rules from separate
        sub-rulesets overlap each other. returns a set of partitions, each a
        set of rules.
        """
        related_rules = dict((rule, self.overlapping_rules(rule)) for rule in self.rules)
        partitions = set()
        while related_rules:
            start = peek(related_rules)
            partition = graph_traverse(related_rules, start)
            partitions.add(set_(partition))
            for rule in partition:
                del related_rules[rule]
        return partitions

    def cells_(self) -> frozenset:
        """return all cells contained in ruleset"""
        return set_(list(self.map.keys()))


class RuleReducer(object):
    """manager object that performs the 'logical deduction' phase of
    the solver; maintains a set of active rules, tracks which rules
    overlap with other rules, and iteratively reduces them until no
    further reductions are possible"""

    def __init__(self) -> None:
        # current list of rules
        self.active_rules: Set[Rule_] = set()
        # reverse lookup for rules containing a given cell
        self.cell_rules_map = CellRulesMap()
        # heap of possible reductions: (priority tuple, sequence, reduction)
        self.candidate_reductions: List[Tuple[Tuple[int, int, float], int, Reduceable]] = []
        self._reduction_seq = 0

    def add_rules(self, rules: Iterable[Rule_]) -> None:
        """add a set of rules to the ruleset"""
        for rule in rules:
            self.add_rule(rule)

    def add_rule(self, rule: Rule_) -> None:
        """add a new rule to the active ruleset"""
        # Inline Rule_.decompose() fast path: most rules are non-trivial.
        if rule.num_mines == 0 or rule.num_mines == rule.num_cells:
            for cell_ in rule.cells_:
                size = len(cell_)
                self.add_base_rule(Rule_(size if rule.num_mines > 0 else 0, set_([cell_]), size))
            # degenerate rules (no cells) disappear here
            return

        self.add_base_rule(rule)

    def add_base_rule(self, rule: Rule_) -> None:
        """helper for adding a rule"""
        if rule in self.active_rules:
            return
        self.active_rules.add(rule)
        self.cell_rules_map.add_rule(rule)
        self.update_reduceables(rule)

    def add_reduceable(self, reduc: Reduceable) -> None:
        # priority queue priorities are lowest first
        prio = tuple(-k for k in reduc.metric())
        heapq.heappush(self.candidate_reductions, (prio, self._reduction_seq, reduc))
        self._reduction_seq += 1

    def update_reduceables(self, rule: Rule_) -> None:
        """update the index of which rules are reduceable from others"""
        for rule_ov in self.cell_rules_map.overlapping_rules(rule):
            if rule_ov.num_cells <= rule.num_cells:
                # Quick mine/non-mine bounds check before subset test.
                if (
                    rule_ov.num_mines <= rule.num_mines
                    and (rule_ov.num_cells - rule_ov.num_mines) <= (rule.num_cells - rule.num_mines)
                    and rule_ov.is_subrule_of(rule)
                ):
                    # catches if rules are equivalent
                    self.add_reduceable(Reduceable(rule, rule_ov))
            elif rule.num_cells < rule_ov.num_cells:
                # Quick mine/non-mine bounds check before subset test.
                if (
                    rule.num_mines <= rule_ov.num_mines
                    and (rule.num_cells - rule.num_mines) <= (rule_ov.num_cells - rule_ov.num_mines)
                    and rule.is_subrule_of(rule_ov)
                ):
                    self.add_reduceable(Reduceable(rule_ov, rule))

    def remove_rule(self, rule: Rule_) -> None:
        """remove a rule from the active ruleset/index, presumably because it
        was reduced"""
        self.active_rules.remove(rule)
        self.cell_rules_map.remove_rule(rule)
        # we can't remove the inner contents of candidate_reductions queue; items
        # are checked for validity when they're popped

    def pop_best_reduction(self) -> Reduceable | None:
        """get the highest-value reduction to perform next"""
        while self.candidate_reductions:
            reduction = heapq.heappop(self.candidate_reductions)[2]
            if not reduction.contained_within(self.active_rules):
                continue
            return reduction
        return None

    def reduce(self, reduction: Reduceable) -> None:
        """perform a reduction"""
        reduced_rule = reduction.reduce()
        self.remove_rule(reduction.superrule)
        self.add_rule(reduced_rule)

    def reduce_all(self) -> Set[Rule_]:
        """run the manager"""
        while True:
            reduction = self.pop_best_reduction()
            if not reduction:
                break
            self.reduce(reduction)
        return self.active_rules


def permute(
    count: int,
    cells: List[frozenset],
    permu: Optional[Set[Tuple[frozenset, int]]] = None,
) -> Iterator[Permutation]:
    """generate all permutations of 'count' mines among 'cells'

    permu -- the sub-permutation in progress, when called as a recursive
        helper function. not actually a Permutation object, but a set of
        (key, value) pairs
    """

    def permu_add(*k: Tuple[frozenset, int]) -> Set[Tuple[frozenset, int]]:
        return (permu or set()).union(k)

    if permu is None:
        permu = set()

    if count == 0:
        yield Permutation(permu_add(*[(cell, 0) for cell in cells]))
    else:
        remaining_size = sum(len(cell) for cell in cells)
        if remaining_size == count:
            yield Permutation(permu_add(*[(cell, len(cell)) for cell in cells]))
        elif remaining_size >= count:
            cell = cells[0]
            for multiplicity in range(min(count, len(cell)), -1, -1):
                for p in permute(count - multiplicity, cells[1:], permu_add((cell, multiplicity))):
                    yield p


class PermutationSet(object):
    """a set of permutations of the same cell set and total # of mines

    may be the full set of possible permutations, or a subset as particular
    permutations are removed due to outside conflicts

    constrained -- False if the set is the full set of possible permutations;
        True if the set has since been reduced; accurate ONLY IF the
        PermutationSet was created with the full set of possibles
    """

    def __init__(self, cells_: frozenset, k: int, permus: Set[Permutation]) -> None:
        """
        cells_ -- set of supercells
        k -- # of mines
        permus -- set of 'Permutation's thereof; all permutations must share
            the same cell set and # of mines! (corresponding to 'cells_' and
            'k')
        """
        self.cells_ = cells_
        self.k = k
        self.permus = permus
        self.constrained = False
        self._compat_cache: Dict[Permutation, Set[Permutation]] = {}
        self._has_compat_cache: Dict[Permutation, bool] = {}

    def _immutable(self):
        """helper function for comparison in unit tests"""
        return (self.cells_, self.k, set_(self.permus))

    @staticmethod
    def from_rule(rule: Rule_) -> PermutationSet:
        """build from all possible permutations of the given rule"""
        return PermutationSet(rule.cells_, rule.num_mines, set(rule.permute()))

    def to_rule(self) -> Rule_:
        """back-construct a Rule_ from this set

        note that the set generated from self.to_rule().from_rule() may not
        match this set, as it cannot account for permutations removed from
        this set due to conflicts"""
        return Rule_(self.k, self.cells_)

    def __iter__(self) -> Iterator[Permutation]:
        """return an iterator over the set of permutations"""
        return self.permus.__iter__()

    def __contains__(self, p):
        """membership test for permutation"""
        return p in self.permus

    def remove(self, permu: Permutation) -> None:
        """remove a permutation from the set, such as if that permutation
        conflicts with another rule"""
        self.permus.remove(permu)
        self.constrained = True
        self._compat_cache.clear()
        self._has_compat_cache.clear()

    def empty(self) -> bool:
        """return whether the set is empty"""
        return not self.permus

    def compatible(self, permu: Permutation) -> PermutationSet:
        """return a new PermutationSet containing only the Permutations that
        are compatible with the given Permutation 'permu'"""
        return PermutationSet(self.cells_, self.k, self.compatible_permus(permu))

    def compatible_permus(self, permu: Permutation) -> Set[Permutation]:
        """return compatible permutations as a plain set for hotpath callers."""
        cached = self._compat_cache.get(permu)
        if cached is not None:
            return cached

        matched = {p for p in self.permus if p.compatible(permu)}
        self._compat_cache[permu] = matched
        self._has_compat_cache[permu] = bool(matched)
        return matched

    def has_compatible(self, permu: Permutation) -> bool:
        """return whether any permutation in this set is compatible with permu."""
        cached = self._has_compat_cache.get(permu)
        if cached is not None:
            return cached

        result = any(p.compatible(permu) for p in self.permus)
        self._has_compat_cache[permu] = result
        return result

    def subset(self, cell_subset: frozenset) -> PermutationSet:
        """return a new PermutationSet consisting of the sub-setted
        permutations from this set"""
        permu_subset = set(p.subset(cell_subset) for p in self.permus)
        k_sub = set(p.k() for p in permu_subset)
        if len(k_sub) > 1:
            # subset is not valid because permutations differ in # of mines
            raise ValueError()
        return PermutationSet(cell_subset, k_sub.pop(), permu_subset)

    def decompose(self) -> list[PermutationSet]:
        """see _decompose(); optimizes if set has not been constrained because
        full permu-sets decompose to themselves"""
        return self._decompose() if self.constrained else [self]

    def _decompose(self, k_floor: int = 1) -> List[PermutationSet]:
        """determine if the permutation set is the cartesian product of N
        smaller permutation sets; return the decomposition if so

        this set may be constrained, in which case at least one subset of the
        decomposition (if one exists) will also be constrained
        """
        for _k in range(k_floor, int(0.5 * len(self.cells_)) + 1):
            for cell_subset in (set_(c) for c in itertools.combinations(self.cells_, _k)):
                try:
                    permu_subset, permu_remainder = self.split(cell_subset)
                except ValueError:
                    continue

                # lo, a cartesian divisor!
                divisors = [permu_subset]
                divisors.extend(permu_remainder._decompose(_k))
                return divisors

        return [self]

    def split(self, cell_subset: frozenset) -> Tuple[PermutationSet, PermutationSet]:
        """helper function for decompose(). given a subset of cells, return
        the two permutation sets for the subset and the set of remaining
        cells, provided cell_subset is a valid decomposor; raise exception if
        not"""
        cell_remainder = self.cells_ - cell_subset
        permu_subset = self.subset(cell_subset)
        # exception thrown if subset cannot be a cartesian divisor; i.e., set
        # of permutations could not have originated from single 'choose'
        # operation

        # get the remaining permutation sets for each sub-permutation
        remainders_by_subset: Dict[Permutation, Set[Permutation]] = {}
        for p in self.permus:
            p_subset = p.subset(cell_subset)
            p_remainder = p.subset(cell_remainder)
            bucket = remainders_by_subset.get(p_subset)
            if bucket is None:
                bucket = set()
                remainders_by_subset[p_subset] = bucket
            bucket.add(p_remainder)

        permu_remainders = set(set_(vals) for vals in remainders_by_subset.values())
        if len(permu_remainders) > 1:
            # remaining subsets are not identical for each sub-permutation; not
            # a cartesian divisor
            raise ValueError()
        permu_remainders = permu_remainders.pop()

        return (
            permu_subset,
            PermutationSet(cell_remainder, self.k - permu_subset.k, set(permu_remainders)),
        )

    def __repr__(self):
        return str(list(self.permus))


class EnumerationState(object):
    """a helper object to enumerate through all possible mine configurations of
    a ruleset"""

    def __init__(self, ruleset: Optional[PermutedRuleset] = None) -> None:
        """
        ruleset -- None when cloning an existing state
        """
        if ruleset is None:
            # 'naked' object for cloning
            return

        # set of Permutations -- one per rule -- that have been 'fixed' for
        # the current configuration-in-progress
        self.fixed: Set = set()
        # subset of ruleset whose permutations are still 'open'
        self.free = {rule: permu_set.permus.copy() for rule, permu_set in ruleset.permu_map.items()}

        # helper function (closure)
        self.overlapping_rules = lambda rule: ruleset.cell_rules_map.overlapping_rules(rule)
        # original permutation sets per rule (shared by clones) for lazy compatibility checks
        self.rule_permuset_map = ruleset.permu_map
        # lazy index for constraining overlapping permutations
        # mapping: (permutation, overlapping rule) -> set of valid permutations for overlapping rule
        self.compatible_rule_index: Dict[Tuple[Permutation, Rule_], Set[Permutation]] = {}

    def clone(self) -> EnumerationState:
        """clone this state"""
        state = EnumerationState()
        state.fixed = set(self.fixed)
        state.free = {rule: permu_set.copy() for rule, permu_set in self.free.items()}
        state.overlapping_rules = self.overlapping_rules
        state.rule_permuset_map = self.rule_permuset_map
        state.compatible_rule_index = self.compatible_rule_index
        return state

    def allowed_permus(self, permu: Permutation, related_rule: Rule_) -> Set[Permutation]:
        """get/set cached compatible permutations for (permu, related_rule)."""
        key = (permu, related_rule)
        allowed = self.compatible_rule_index.get(key)
        if allowed is None:
            allowed = self.rule_permuset_map[related_rule].compatible_permus(permu)
            self.compatible_rule_index[key] = allowed
        return allowed

    def is_complete(self) -> bool:
        """return whether all rules have been 'fixed', i.e., the configuration
        is complete"""
        return not self.free

    def __iter__(self) -> Iterator[EnumerationState]:
        """pick an 'open' rule at random and 'fix' each possible permutation
        for that rule. in this manner, when done recursively, all valid
        combinations are enumerated"""
        rule = min(self.free, key=lambda r: len(self.free[r]))
        for permu in self.free[rule]:
            try:
                yield self.propogate(rule, permu)
            except ValueError:
                # conflict detected; dead end
                pass

    def propogate(self, rule: Rule_, permu: Permutation) -> EnumerationState:
        """'fix' a permutation for a given rule"""
        state = self.clone()
        state._propogate(rule, permu)
        return state

    def _propogate(self, rule: Rule_, permu: Permutation) -> None:
        """'fix' a rule permutation and constrain the available permutations
        of all overlapping rules"""
        cascades = [(rule, permu)]
        while cascades:
            curr_rule, curr_permu = cascades.pop()
            if curr_rule not in self.free:
                # May already have been constrained by a prior cascade.
                continue

            self.fixed.add(curr_permu)
            del self.free[curr_rule]

            for related_rule in self.overlapping_rules(curr_rule):
                linked_permus = self.free.get(related_rule)
                if linked_permus is None:
                    continue

                # PermutationSet of related_rule constrained only by curr_rule/curr_permu.
                allowed_permus = self.allowed_permus(curr_permu, related_rule)
                linked_permus.intersection_update(allowed_permus)

                if not linked_permus:
                    # conflict
                    raise ValueError()
                if len(linked_permus) == 1:
                    # only one possibility; constrain further
                    cascades.append((related_rule, peek(linked_permus)))

    def mine_config(self) -> Permutation:
        """convert the set of fixed permutations into a single Permutation
        encompassing the mine configuration for the entire ruleset"""
        merged: dict = {}
        for permu in self.fixed:
            merged.update(permu.mapping)
        return Permutation(merged)

    def enumerate(self) -> Iterator[Permutation]:
        """recursively generate all possible mine configurations for the ruleset"""
        if self.is_complete():
            yield self.mine_config()
        else:
            for next_state in self:
                for mineconfig in next_state.enumerate():
                    yield mineconfig


class FrontSubtally(object):
    """sub-tabulation of per-cell mine frequencies"""

    def __init__(self) -> None:
        # 'weight' of this sub-tally among the others in the FrontTally. initially
        # will be a raw count of the configurations in this sub-tally, but later
        # will be skewed due to weighting and normalizing factors
        self.total: float = 0
        # per-cell mine counts (pre-finalizing) / mine prevalence (post-finalizing)
        # mapping: supercell -> total # of mines in cell summed across all configurations (pre-finalize)
        #                    -> expected # of mines in cell (post-finalize)
        self.tally: Dict = collections.defaultdict(lambda: 0)

        self.finalized = False
        self.normalized = False

    def add(self, config: Permutation) -> None:
        """add a configuration to the tally"""
        mult = config.multiplicity()  # weight by multiplicity
        self.total += mult
        for cell_, n in config.mapping.items():
            self.tally[cell_] += n * mult

    def finalize(self) -> None:
        """after all configurations have been summed, compute relative
        prevalence from totals"""
        self.tally = dict((cell_, n / float(self.total)) for cell_, n in self.tally.items())
        self.finalized = True

    def collapse(
        self,
    ) -> Iterator[Tuple[Any, Any]]:
        """helper function for FrontTally.collapse(); emit all cell expected
        mine values weighted by this sub-tally's weight"""
        for cell_, expected_mines in self.tally.items():
            yield (cell_, self.total * expected_mines)

    @staticmethod
    def mk(total: int | float, tally: Any) -> FrontSubtally:
        """build a sub-tally manually

        tally data must already be finalized"""
        o = FrontSubtally()
        o.total = total
        o.tally = tally
        o.finalized = True
        return o

    def __repr__(self):
        return str((self.total, dict(self.tally)))


def enumerate_front(front: PermutedRuleset) -> FrontTally:
    """enumerate and tabulate all mine configurations for the given front

    return a tally where: sub-totals are split out by total # of mines in
    configuration, and each sub-tally contains: a total count of matching
    configurations, and expected # of mines in each cell
    """
    tally = FrontTally()
    tally.tally(front)
    return tally


def cell_probabilities(
    tallies: Set[FrontTally],
    mine_prevalence: MineCount | float,
    all_cells: List[frozenset],
) -> Iterator[Tuple[Any, float]]:
    """generate the final expected values for all cells in all fronts

    tallies -- set of 'FrontTally's
    mine_prevalence -- description of # or frequency of mines in board
        (from solve())
    all_cells -- a set of all supercells from all rules

    generates a stream of tuples: (cell, # mines / cell) for all cells
    """

    tally_uncharted = weight_subtallies(tallies, mine_prevalence, all_cells)
    # concatenate and emit the cell solutions from all fronts
    return itertools.chain(*(tally.collapse() for tally in tallies), tally_uncharted.collapse())


def weight_subtallies(
    tallies: Set[FrontTally], mine_prevalence: MineCount | float, all_cells: List[frozenset]
) -> FrontTally | FixedProbTally:
    """analyze all FrontTallys as a whole and weight the likelihood of each
    sub-tally using probability analysis"""

    # True: traditional minesweeper -- fixed total # of mines
    # False: fixed overall probability of mine; total # of mines varies per game
    discrete_mode = isinstance(mine_prevalence, MineCount)

    if discrete_mode:
        num_uncharted_cells = check_count_consistency(tallies, mine_prevalence, all_cells)

    # tallies with only one sub-tally don't need weighting
    dyn_tallies = set(tally for tally in tallies if not tally.is_static())

    if discrete_mode:
        num_static_mines = sum(tally.max_mines() for tally in (tallies - dyn_tallies))
        at_large_mines = mine_prevalence.total_mines - num_static_mines

        tally_uncharted = combine_fronts(dyn_tallies, num_uncharted_cells, at_large_mines)
    else:
        tally_uncharted = weight_nondiscrete(dyn_tallies, mine_prevalence)
    return tally_uncharted


def weight_nondiscrete(dyn_tallies: Set[FrontTally], mine_prevalence: float) -> FixedProbTally:
    """weight the relative likelihood of each sub-tally in a 'fixed mine
    probability / variable # of mines'-style game

    in this scenario, all fronts are completely independent; no front affects
    the likelihoods for any other front
    """
    for tally in dyn_tallies:
        relative_likelihood = lambda num_mines: nondiscrete_relative_likelihood(mine_prevalence, num_mines, tally.min_mines())
        tally.scale_weights(relative_likelihood)

    # regurgitate the fixed mine probability as the p for 'other' cells. kind of
    # redundant but saves the client a step. (note that since we don't count total
    # # cells in this mode, this is not a guarantee that any given game state has
    # 'other' cells)
    return FixedProbTally(mine_prevalence)


def check_count_consistency(tallies: Set[FrontTally], mine_prevalence: MineCount, all_cells: List[frozenset]) -> int:
    """ensure the min/max mines required across all fronts is compatible with
    the total # of mines and remaining space available on the board

    in the process, compute and return the remaining available space (# of
    'other' cells not referenced in any rule)
    """

    min_possible_mines, max_possible_mines = possible_mine_limits(tallies)
    num_uncharted_cells = mine_prevalence.total_cells - sum(len(cell_) for cell_ in all_cells)

    if min_possible_mines > mine_prevalence.total_mines:
        raise InconsistencyError("minimum possible number of mines is more than supplied mine count")
    if mine_prevalence.total_mines > max_possible_mines + num_uncharted_cells:
        # the max # of mines that can fit on the board is less than the total # specified
        raise InconsistencyError("maximum possible number of mines on board is less than supplied mine count")

    return num_uncharted_cells


class FrontPerMineTotals(object):
    """object that tracks, for a constituent front, how many configurations for each # of mines
    in the front"""

    def __init__(self, totals: Mapping[int, float] | Mapping[int, int]) -> None:
        """totals: mapping of # mines -> # configurations"""
        self.totals = dict((num_mines, float(total)) for num_mines, total in totals.items())

    @staticmethod
    def singleton(num_mines: int, total: float) -> FrontPerMineTotals:
        return FrontPerMineTotals({num_mines: total})

    @property
    def total_count(self) -> float:
        """returns total # of configurations across all possible # of mines"""
        return sum(self.totals.values())

    def multiply(self, n: float) -> FrontPerMineTotals:
        """multiply all the configuration counts by a fixed factor"""
        return FrontPerMineTotals(dict((num_mines, n * count) for num_mines, count in self))

    @staticmethod
    def sum(front_totals: Tuple[FrontPerMineTotals, ...]) -> FrontPerMineTotals:
        """compute an aggregate sum of several mappings"""
        totals: Dict[int, float] = {}
        for ft in front_totals:
            for num_mines, total in ft:
                totals[num_mines] = totals.get(num_mines, 0.0) + total
        return FrontPerMineTotals(totals)

    def __iter__(self) -> Iterator[tuple[int, float]]:
        return iter(self.totals.items())

    def __repr__(self):
        return str(self.totals)


class AllFrontsPerMineTotals(object):
    """object that tracks, for a given # of mines in the CombinedFront, the FrontPerMineTotals objects
    corresponding to each constituent front"""

    def __init__(self, front_totals: List[FrontPerMineTotals]) -> None:
        """front_totals: a list of FrontPerMineTotals objects"""
        self.front_totals = front_totals

    @property
    def total_count(self) -> float:
        """total number of configurations for the given total # of mines in the combined front"""
        if self.front_totals:
            # the count should match for each constituent front
            return self.front_totals[0].total_count
        else:
            # null case
            return 1

    @staticmethod
    def null() -> AllFrontsPerMineTotals:
        return AllFrontsPerMineTotals([])

    @staticmethod
    def singleton(num_mines: int, total: float) -> AllFrontsPerMineTotals:
        return AllFrontsPerMineTotals([FrontPerMineTotals.singleton(num_mines, total)])

    def join_with(self, new: Self) -> AllFrontsPerMineTotals:
        """merge two AllFrontsPerMineTotals objects, joining into a single list and performing
        necessary cross-multiplication"""
        return AllFrontsPerMineTotals(
            [f.multiply(new.total_count) for f in self.front_totals] + [f.multiply(self.total_count) for f in new.front_totals]
        )

    @staticmethod
    def sum(frontsets: List[AllFrontsPerMineTotals]) -> AllFrontsPerMineTotals:
        """sum a list of AllFrontsPerMineTotals objects on a per-constituent front basis"""
        return AllFrontsPerMineTotals(list(map(FrontPerMineTotals.sum, list(zip(*frontsets)))))

    def __iter__(self) -> Iterator[FrontPerMineTotals]:
        return iter(self.front_totals)

    def __repr__(self):
        return str((self.total_count, self.front_totals))


class CombinedFront(object):
    """a representation of a combinatorial fusing of multiple fronts/tallies. essentially, track:
    for total # of mines in the combined front -> for each constituent front -> count of total # of
    configurations for each # of mines in the constituent front"""

    def __init__(self, total_mines_to_front_totals: Dict[int, AllFrontsPerMineTotals]) -> None:
        """total_mines_to_front_totals: a mapping of total # of mines in the combined front to
        a AllFrontsPerMineTotals object"""
        self.totals = total_mines_to_front_totals

    @property
    def min_max_mines(self) -> Tuple[int, int]:
        """return (min, max) # of mines in the front"""
        keys = list(self.totals.keys())
        return (min(keys), max(keys))

    @staticmethod
    def null() -> CombinedFront:
        """create an 'empty' combined front"""
        return CombinedFront({0: AllFrontsPerMineTotals.null()})

    @staticmethod
    def from_counts_per_num_mines(mines_with_count: Iterator) -> CombinedFront:
        """build a starter combined front using known counts for each # of mines"""
        return CombinedFront(
            dict((num_mines, AllFrontsPerMineTotals.singleton(num_mines, total)) for num_mines, total in mines_with_count)
        )

    @staticmethod
    def from_tally(tally: FrontTally) -> CombinedFront:
        """build a starter combined front from a front tally"""
        return CombinedFront.from_counts_per_num_mines((num_mines, subtally.total) for num_mines, subtally in tally)

    @staticmethod
    def for_other(
        min_mines: Union[int, int],
        max_mines: Union[int, int],
        num_uncharted_cells: int,
        max_other_mines: int,
    ) -> CombinedFront:
        """build a starter combined front to represent the 'uncharted cells' region"""
        return CombinedFront.from_counts_per_num_mines(
            (n, relative_likelihood(n, num_uncharted_cells, max_other_mines)) for n in range(min_mines, max_mines + 1)
        )

    # @staticmethod
    def join_with(
        self,
        new: Self,
        min_remaining_mines: int,
        max_remaining_mines: int,
        at_large_mines: int,
    ) -> CombinedFront:
        """combine two combined fronts. min/max remaining mines represent the total remaining mines available
        in all fronts yet to be combined (excluding 'new'). this helps avoid computing combinations whose # mines
        can never add up to the requisite # of board mines. this is also how we converge to a single total # of
        mines upon combining all fronts"""

        def cross_entry(
            pair: Tuple[Tuple[int, AllFrontsPerMineTotals], Tuple[int, AllFrontsPerMineTotals]]
        ) -> Tuple[int, AllFrontsPerMineTotals] | None:
            ((a_num_mines, a_fronts), (b_num_mines, b_fronts)) = pair
            combined_mines = a_num_mines + b_num_mines
            min_mines_at_end = combined_mines + min_remaining_mines
            max_mines_at_end = combined_mines + max_remaining_mines
            if min_mines_at_end > at_large_mines or max_mines_at_end < at_large_mines:
                return None
            return (combined_mines, a_fronts.join_with(b_fronts))

        grouped: Dict[int, List[AllFrontsPerMineTotals]] = {}
        for pair in itertools.product(self, new):
            entry = cross_entry(pair)
            if not entry:
                continue
            num_mines, fronts = entry
            bucket = grouped.get(num_mines)
            if bucket is None:
                bucket = []
                grouped[num_mines] = bucket
            bucket.append(fronts)

        new_totals = dict((num_mines, AllFrontsPerMineTotals.sum(frontsets)) for num_mines, frontsets in grouped.items())
        return CombinedFront(new_totals)

    def collapse(self) -> List[dict[int, float] | dict[int, int]]:
        """once all fronts combined, unwrap objects and return the underlying counts corresponding to each front"""
        assert len(self.totals) == 1
        return [e.totals for e in self.totals.popitem()[1].front_totals]

    def __iter__(self) -> Iterator[tuple[int, AllFrontsPerMineTotals]]:
        return iter(self.totals.items())

    def __repr__(self):
        return str(self.totals)


def relative_likelihood(num_free_mines: int, num_uncharted_cells: int, max_other_mines: int) -> float:
    return discrete_relative_likelihood(num_uncharted_cells, num_free_mines, max_other_mines)


def combine_fronts(
    tallies: Set[FrontTally],
    num_uncharted_cells: int,
    at_large_mines: int,
) -> FrontTally:
    """assign relative weights to all sub-tallies in all fronts. because the
    total # of mines is fixed, we must do a combinatorial analysis to
    compute the likelihood of each front containing each possible # of mines.
    in the process, compute the mine count likelihood for the 'other' cells,
    not a part of any front, and return a meta-front encapsulating them.
    """

    min_tallied_mines, max_tallied_mines = possible_mine_limits(set(tallies))
    min_other_mines = max(at_large_mines - max_tallied_mines, 0)
    # technically, min_tallied_mines known to be <= at_large_mines due to check_count_consistency()
    max_other_mines = min(max(at_large_mines - min_tallied_mines, 0), num_uncharted_cells)

    tallies = set(tallies)  # we need guaranteed iteration order
    all_fronts: list[CombinedFront] = list(map(CombinedFront.from_tally, tallies)) + [
        CombinedFront.for_other(min_other_mines, max_other_mines, num_uncharted_cells, max_other_mines)
    ]
    min_remaining_mines, max_remaining_mines = list(map(sum, list(zip(*(f.min_max_mines for f in all_fronts)))))
    combined: CombinedFront = CombinedFront.null()
    for f in all_fronts:
        # note that it's only safe to use min/max mines in this way before the front has been combined/modified
        front_min, front_max = f.min_max_mines
        min_remaining_mines -= front_min
        max_remaining_mines -= front_max
        combined = combined.join_with(f, min_remaining_mines, max_remaining_mines, at_large_mines)
    front_totals = combined.collapse()
    uncharted_total = front_totals[-1]
    front_totals = front_totals[:-1]

    # upate tallies with adjusted weights
    for tally, front_total in zip(tallies, front_totals):
        tally.update_weights(front_total)

    return FrontTally.for_other(num_uncharted_cells, uncharted_total)


def possible_mine_limits(tallies: Set[FrontTally]) -> Tuple[int, int]:
    """return the total minimum and maximum possible # of mines across all
    tallied fronts

    returns (min, max)"""
    return (
        sum(tally.min_mines() for tally in tallies),
        sum(tally.max_mines() for tally in tallies),
    )


def nondiscrete_relative_likelihood(p, k, k0):
    """given binomial probability (p,k,n) => p^k*(1-p)^(n-k),
    return binom_prob(p,k,n) / binom_prob(p,k0,n)

    note that n isn't actually needed! this is because we're calculating a
    per-configuration weight, and in a true binomial distribution we'd then
    multiply by (n choose k) configurations; however, we've effectively done
    that already with the enumeration/tallying phase
    """

    if p < 0.0 or p > 1.0:
        raise ValueError("p must be [0., 1.]")

    return float((p / (1 - p)) ** (k - k0))


def discrete_relative_likelihood(n: int, k: int, k0: int) -> float:
    """return 'n choose k' / 'n choose k0'"""
    if any(x < 0 or x > n for x in (k, k0)):
        raise ValueError("k, k0 must be [0, n]")

    return float(fact_div(k0, k) * fact_div(n - k0, n - k))


class FixedProbTally(ImmutableMixin):
    """a meta-tally to represent when all 'other' cells are uncounted and
    assumed to have a fixed mine probability"""

    def __init__(self, p):
        self.p = p

    def collapse(self):
        yield (UnchartedCell(), self.p)

    def _canonical(self):
        return (self.p,)


def expand_cells(cell_probs: Iterator[Tuple[Any, float]], other_tag: Optional[Any]) -> Iterator[Tuple[Any, float]]:
    """back-convert the expected values for all supercells into per-cell
    probabilities for each original cell"""
    for cell_, p in cell_probs:
        for cell in cell_:
            yield (cell if cell is not None else other_tag, p / len(cell_))


def permute_and_interfere(rules: Set[Rule_]) -> PermutedRuleset:
    """process the set of rules and analyze the relationships and constraints
    among them"""
    ruleset = PermutedRuleset(rules)
    ruleset.cross_eliminate()
    ruleset.rereduce()
    return ruleset


def reduce_rules(rules: List[Rule_]) -> Set[Rule_]:
    """reduce ruleset using logical deduction"""
    if not rules:
        return set()
    rr = RuleReducer()
    rr.add_rules(set(rules))
    return rr.reduce_all()


def condense_supercells(rules: Set[Rule]) -> Tuple[List[Rule_], List[frozenset]]:
    """condense supercells by finding sets of ordinary cells that only ever
    appear together. returns a set of 'Rule_' corresponding to the original
    ruleset.

    rules -- original set of 'Rule' to analyze

    note that ALL cells are converted to supercells for ease of processing
    later, even if that cell does not group with any others. the result would
    be a singleton supercell
    """
    if not rules:
        return ([], [])

    # for each cell, list of rules that cell appears in
    cell_rules_tmp: Dict[Any, Set[Rule]] = collections.defaultdict(set)
    for rule in rules:
        for cell in rule.cells:
            cell_rules_tmp[cell].add(rule)

    # for each 'list of rules appearing in', list of cells that share that ruleset (these cells
    # thus only ever appear together in the same rules)
    rules_supercell_tmp: Dict[frozenset, Set[Any]] = collections.defaultdict(set)
    for cell, cell_rules in cell_rules_tmp.items():
        rules_supercell_tmp[set_(cell_rules)].add(cell)

    rules_supercell_map: Dict[frozenset, frozenset] = dict(
        (ruleset, set_(cells)) for ruleset, cells in rules_supercell_tmp.items()
    )

    # for each original rule, list of 'supercells' appearing in that rule
    rule_supercells_tmp: Dict[Rule, Set[frozenset]] = collections.defaultdict(set)
    for ruleset, supercell in rules_supercell_map.items():
        for rule in ruleset:
            rule_supercells_tmp[rule].add(supercell)

    rule_supercells_map: Dict[Rule, frozenset] = dict(
        (rule, set_(supercells)) for rule, supercells in rule_supercells_tmp.items()
    )

    return (
        [rule.condensed(rule_supercells_map) for rule in rules],
        list(rules_supercell_map.values()),
    )


def solve(
    rs: Set[Rule], mine_prevalence: MineCount | float, other_tag: Optional[Any] = None
) -> Dict[Any, float]:
    """solve a minesweeper board.

    take in a minesweeper board and return the solution as a dict mapping each
    cell to its probability of being a mine.

    rs -- a set of 'Rule' describing the board
    mine_prevalence -- an object describing the total expected mines on the
        board. a MineCount indicates traditional minesweeper (fixed board
        dimensions with a total # of mines); a float indicates a fixed
        probability that any unknown cell is a mine (total # of mines will
        vary for given board dimensions, in a binomial distribution)
    other_tag -- tag used to represent all 'other' cells (all cells not
        mentioned in a rule) in the solution output
    """
    rules, all_cells = condense_supercells(rs)
    ruless = reduce_rules(rules)

    determined = set(r for r in ruless if r.is_trivial())
    ruless -= determined

    fronts: Set[PermutedRuleset] = set()
    if ruless:
        fronts = permute_and_interfere(ruless).split_fronts()

        trivial_fronts = set(f for f in fronts if f.is_trivial())
        if trivial_fronts:
            determined |= set(f.trivial_rule() for f in trivial_fronts)
            fronts -= trivial_fronts

    stats = set(enumerate_front(f) for f in fronts)
    stats.update(r.tally() for r in determined)
    cell_probs = cell_probabilities(stats, mine_prevalence, all_cells)
    return dict(expand_cells(cell_probs, other_tag))
