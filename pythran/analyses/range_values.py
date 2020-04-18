""" Module Analysing code to extract positive subscripts from code.  """
# TODO check bound of while and if for more accurate values.

import gast as ast
from collections import defaultdict
from functools import reduce

from pythran.analyses import Aliases
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes


def combine(op, node0, node1):
    key = '__{}__'.format(op.__class__.__name__.lower())
    try:
        return getattr(type(node0), key)(node0, node1)
    except AttributeError:
        return UNKNOWN_RANGE


def bound_range(mapping, node, modified=None):
    if modified is None:
        modified = set()

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value in node.values:
                bound_range(mapping, value, modified)
        elif isinstance(node.op, ast.Or):
            mappings = [mapping.copy() for _ in node.values]
            for value, mapping_cpy in zip(node.values, mappings):
                bound_range(mapping_cpy, value, modified)
            for k in modified:
                mapping[k] = reduce(lambda x, y: x.union(y[k]),
                                    mappings[1:],
                                    mappings[0][k])

    if isinstance(node, ast.Compare):
        current_bound = None
        current_identifiers = []
        if isinstance(node.left, ast.Name):
            current_identifiers.append(node.left.id)
            modified.add(node.left.id)
        elif isinstance(getattr(node.left, 'value', None), (bool, int)):
            current_bound = node.left.value

        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(comparator, ast.Name):
                current_identifiers.append(comparator.id)
                modified.add(comparator.id)

                if current_bound is None:
                    continue
                if isinstance(op, ast.Eq):
                    interval = Interval(current_bound, current_bound)
                elif isinstance(op, ast.Lt):
                    interval = Interval(current_bound + 1, float('inf'))
                elif isinstance(op, ast.LtE):
                    interval = Interval(current_bound, float('inf'))
                elif isinstance(op, ast.Gt):
                    interval = Interval(-float('inf'), current_bound - 1)
                elif isinstance(op, ast.GtE):
                    interval = Interval(float('inf'), current_bound)
                else:
                    interval = None

            elif isinstance(comparator, (ast.List, ast.Tuple, ast.Set)):
                if all(isinstance(getattr(elt, 'value', None), (bool, int))
                       for elt in comparator.elts):
                    if isinstance(op, ast.In) and comparator.elts:
                        low = min(elt.value for elt in comparator.elts)
                        high = max(elt.value for elt in comparator.elts)
                        interval = Interval(low, high)
                    else:
                        interval = None
                else:
                    interval = None

            elif isinstance(getattr(comparator, 'value', None), (bool, int)):
                current_bound = comparator.value
                if isinstance(op, ast.Eq):
                    interval = Interval(current_bound, current_bound)
                elif isinstance(op, ast.Lt):
                    interval = Interval(-float('inf'), current_bound - 1)
                elif isinstance(op, ast.LtE):
                    interval = Interval(float('inf'), current_bound)
                elif isinstance(op, ast.Gt):
                    interval = Interval(current_bound + 1, float('inf'))
                elif isinstance(op, ast.GtE):
                    interval = Interval(current_bound, float('inf'))
                else:
                    interval = None
            else:
                interval = None

            if interval is not None:
                for name in current_identifiers:
                    mapping[name] = mapping[name].intersect(interval)


class RangeValues(ModuleAnalysis):

    """
    This analyse extract positive subscripts from code.

    It is flow sensitive and aliasing is not taken into account as integer
    doesn't create aliasing in Python.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse('''
    ... def foo(a):
    ...     for i in builtins.range(1, 10):
    ...         c = i // 2''')
    >>> pm = passmanager.PassManager("test")
    >>> res = pm.gather(RangeValues, node)
    >>> res['c'], res['i']
    (Interval(low=0, high=5), Interval(low=1, high=10))
    """

    ResultHolder = object()

    def __init__(self):
        """Initialize instance variable and gather globals name information."""
        self.result = defaultdict(lambda: UNKNOWN_RANGE)
        super(RangeValues, self).__init__(Aliases)

    def add(self, variable, range_):
        """
        Add a new low and high bound for a variable.

        As it is flow insensitive, it compares it with old values and update it
        if needed.
        """
        if variable not in self.result:
            self.result[variable] = range_
        else:
            self.result[variable] = self.result[variable].union(range_)
        return self.result[variable]

    def visit_FunctionDef(self, node):
        """ Set default range value for globals and attributes.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(a, b): pass")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=-inf, high=inf)
        """
        if node in self.result:
            return
        self.result[node] = UNKNOWN_RANGE

        prev_result = self.result.get(RangeValues.ResultHolder, None)

        # Set this prematurely to avoid infinite callgraph loop

        for stmt in node.body:
            self.visit(stmt)

        del self.result[node]
        self.add(node, self.result[RangeValues.ResultHolder])

        if prev_result is not None:
            self.result[RangeValues.ResultHolder] = prev_result

    def visit_Return(self, node):
        if node.value:
            return_range = self.visit(node.value)
            return self.add(RangeValues.ResultHolder, return_range)
        else:
            return self.generic_visit(node)

    def visit_Assert(self, node):
        """
        Constraint the range of variables

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(a): assert a >= 1; b = a + 1")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=1, high=inf)
        >>> res['b']
        Interval(low=2, high=inf)
        """
        bound_range(self.result, node.test)

    def visit_Assign(self, node):
        """
        Set range value for assigned variable.

        We do not handle container values.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(): a = b = 2")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=2, high=2)
        >>> res['b']
        Interval(low=2, high=2)
        """
        assigned_range = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Make sure all Interval doesn't alias for multiple variables.
                self.add(target.id, assigned_range)
            else:
                self.visit(target)

    def visit_AugAssign(self, node):
        """ Update range value for augassigned variables.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(): a = 2; a -= 1")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=1, high=2)
        """
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            name = node.target.id
            res = combine(node.op,
                          self.result[name],
                          self.result[node.value])
            self.result[name] = self.result[name].union(res)

    def visit_For(self, node):
        """ Handle iterate variable in for loops.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = b = c = 2
        ...     for i in builtins.range(1):
        ...         a -= 1
        ...         b += 1''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=-inf, high=2)
        >>> res['b']
        Interval(low=2, high=inf)
        >>> res['c']
        Interval(low=2, high=2)

        >>> node = ast.parse('''
        ... def foo():
        ...     for i in (1, 2, 4):
        ...         a = i''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=1, high=4)
        """
        assert isinstance(node.target, ast.Name), "For apply on variables."
        self.visit(node.iter)
        if isinstance(node.iter, ast.Call):
            for alias in self.aliases[node.iter.func]:
                if isinstance(alias, Intrinsic):
                    self.add(node.target.id,
                             alias.return_range_content(
                                 [self.visit(n) for n in node.iter.args]))

        self.visit_loop(node,
                        ast.Compare(node.target, [ast.In()], [node.iter]))

    def visit_loop(self, node, cond=None):
        """ Handle incremented variables in loop body.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = b = c = 2
        ...     while a > 0:
        ...         a -= 1
        ...         b += 1''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=0, high=2)
        >>> res['b']
        Interval(low=2, high=inf)
        >>> res['c']
        Interval(low=2, high=2)
        """

        if cond is not None:
            init_range = self.result
            self.result = self.result.copy()
            bound_range(self.result, cond)

        # visit once to gather newly declared vars
        for stmt in node.body:
            self.visit(stmt)

        # freeze current state
        old_range = self.result.copy()

        # extra round
        for stmt in node.body:
            self.visit(stmt)

        # widen any change
        for expr, range_ in old_range.items():
            self.result[expr] = self.result[expr].widen(range_)

        # propagate the new informations again
        if cond is not None:
            bound_range(self.result, cond)
            for stmt in node.body:
                self.visit(stmt)
            for k, v in init_range.items():
                self.result[k] = self.result[k].union(v)
            self.visit(cond)

        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node):
        self.visit(node.test)
        return self.visit_loop(node, node.test)

    def visit_BoolOp(self, node):
        """ Merge right and left operands ranges.

        TODO : We could exclude some operand with this range information...

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2
        ...     c = 3
        ...     d = a or c''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['d']
        Interval(low=2, high=3)
        """
        res = list(zip(*[self.visit(elt).bounds() for elt in node.values]))
        return self.add(node, Interval(min(res[0]), max(res[1])))

    def visit_BinOp(self, node):
        """ Combine operands ranges for given operator.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2
        ...     c = 3
        ...     d = a - c''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['d']
        Interval(low=-1, high=-1)
        """
        res = combine(node.op, self.visit(node.left), self.visit(node.right))
        return self.add(node, res)

    def visit_UnaryOp(self, node):
        """ Update range with given unary operation.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2
        ...     c = -a
        ...     d = ~a
        ...     f = +a
        ...     e = not a''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['f']
        Interval(low=2, high=2)
        >>> res['c']
        Interval(low=-2, high=-2)
        >>> res['d']
        Interval(low=-3, high=-3)
        >>> res['e']
        Interval(low=0, high=1)
        """
        res = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            res = Interval(0, 1)
        elif(isinstance(node.op, ast.Invert) and
             isinstance(res.high, int) and
             isinstance(res.low, int)):
            res = Interval(~res.high, ~res.low)
        elif isinstance(node.op, ast.UAdd):
            pass
        elif isinstance(node.op, ast.USub):
            res = Interval(-res.high, -res.low)
        else:
            res = UNKNOWN_RANGE
        return self.add(node, res)

    def visit_If(self, node):
        """ Handle iterate variable across branches

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> pm = passmanager.PassManager("test")

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = 1
        ...     else: b = 3''')

        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=2, high=inf)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if 0 < a < 4: b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (0 < a) and (a < 4): b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (a == 1) or (a == 2): b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)
        """
        self.visit(node.test)
        old_range = self.result

        self.result = old_range.copy()
        bound_range(self.result, node.test)

        for stmt in node.body:
            self.visit(stmt)
        body_range = self.result

        self.result = old_range.copy()
        for stmt in node.orelse:
            self.visit(stmt)
        orelse_range = self.result

        self.result = body_range
        for k, v in orelse_range.items():
            if k in self.result:
                self.result[k] = self.result[k].union(v)
            else:
                self.result[k] = v

    def visit_IfExp(self, node):
        """ Use worst case for both possible values.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2 or 3
        ...     b = 4 or 5
        ...     c = a if a else b''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['c']
        Interval(low=2, high=5)
        """
        self.visit(node.test)
        body_res = self.visit(node.body)
        orelse_res = self.visit(node.orelse)
        return self.add(node, orelse_res.union(body_res))

    def visit_Compare(self, node):
        """ Boolean are possible index.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2 or 3
        ...     b = 4 or 5
        ...     c = a < b
        ...     d = b < 3
        ...     e = b == 4''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['c']
        Interval(low=1, high=1)
        >>> res['d']
        Interval(low=0, high=0)
        >>> res['e']
        Interval(low=0, high=1)
        """
        if any(isinstance(op, (ast.In, ast.NotIn, ast.Is, ast.IsNot))
               for op in node.ops):
            self.generic_visit(node)
            return self.add(node, Interval(0, 1))

        curr = self.visit(node.left)
        res = []
        for op, comparator in zip(node.ops, node.comparators):
            comparator = self.visit(comparator)
            fake = ast.Compare(ast.Name('x', ast.Load(), None, None),
                               [op],
                               [ast.Name('y', ast.Load(), None, None)])
            fake = ast.Expression(fake)
            ast.fix_missing_locations(fake)
            expr = compile(ast.gast_to_ast(fake), '<range_values>', 'eval')
            res.append(eval(expr, {'x': curr, 'y': comparator}))
        if all(res):
            return self.add(node, Interval(1, 1))
        elif any(r.low == r.high == 0 for r in res):
            return self.add(node, Interval(0, 0))
        else:
            return self.add(node, Interval(0, 1))

    def visit_Call(self, node):
        """ Function calls are not handled for now.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = builtins.range(10)''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=-inf, high=inf)
        """
        for alias in self.aliases[node.func]:
            if alias is MODULES['builtins']['getattr']:
                attr_name = node.args[-1].value
                attribute = attributes[attr_name][-1]
                self.add(node, attribute.return_range(None))
            elif isinstance(alias, Intrinsic):
                alias_range = alias.return_range(
                    [self.visit(n) for n in node.args])
                self.add(node, alias_range)
            elif isinstance(alias, ast.FunctionDef):
                if alias not in self.result:
                    self.visit(alias)
                self.add(node, self.result[alias])
            else:
                self.result.pop(node, None)
                return self.generic_visit(node)
        return self.result[node]

    def visit_Constant(self, node):
        """ Handle literals integers values. """
        if isinstance(node.value, (bool, int)):
            return self.add(node, Interval(node.value, node.value))
        return UNKNOWN_RANGE

    def visit_Name(self, node):
        """ Get range for parameters for examples or false branching. """
        return self.add(node, self.result[node.id])

    def visit_Tuple(self, node):
        return self.add(node,
                        IntervalTuple(self.visit(elt) for elt in node.elts))

    def visit_Index(self, node):
        return self.add(node, self.visit(node.value))

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Call):
            for alias in self.aliases[node.value.func]:
                if alias is MODULES['builtins']['getattr']:
                    attr_name = node.value.args[-1].value
                    attribute = attributes[attr_name][-1]
                    self.add(node, attribute.return_range_content(None))
                elif isinstance(alias, Intrinsic):
                    self.add(node,
                             alias.return_range_content(
                                 [self.visit(n) for n in node.value.args]))
                else:
                    return self.generic_visit(node)
            if not self.aliases[node.value.func]:
                return self.generic_visit(node)
            self.visit(node.slice)
            return self.result[node]
        else:
            value = self.visit(node.value)
            slice = self.visit(node.slice)
            return self.add(node, value[slice])

    def visit_ExceptHandler(self, node):
        """ Add a range value for exception variable.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     try:
        ...         pass
        ...     except builtins.RuntimeError as e:
        ...         pass''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['e']
        Interval(low=-inf, high=inf)
        """
        for stmt in node.body:
            self.visit(stmt)

    def generic_visit(self, node):
        """ Other nodes are not known and range value neither. """
        super(RangeValues, self).generic_visit(node)
        return self.add(node, UNKNOWN_RANGE)
