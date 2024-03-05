from typing import Iterable


class SetList(list):
    def __init__(self, obj: Iterable = None):
        super().__init__()
        if obj is not None:
            self.extend(obj)

    def append(self, obj) -> None:
        if obj not in self:
            super().append(obj)

    def extend(self, iterable: Iterable) -> None:
        for e in iterable:
            if e not in self:
                super().append(e)

    def insert(self, index, obj) -> None:
        if obj not in self:
            super().insert(index, obj)
        else:
            raise ValueError('%s is already present in SetList object.' % str(obj))

    def replace(self, old_value, new_value):
        # WARNING: To be used only when `new_value` is not already present in SetList
        self[self.index(old_value)] = new_value

    def __eq__(self, other):
        if isinstance(other, SetList):
            return set(self) == set(other)
        return False

    def __add__(self, other):
        res = SetList(self)
        res.extend(other)
        return res

    def __sub__(self, other):
        return SetList([x for x in self if x not in other])

    def __setitem__(self, key, value):
        if value not in self:
            super().__setitem__(key, value)
        else:
            raise ValueError('%s is already present in SetList object.' % str(value))

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __mul__(self, other):
        raise ValueError('Cannot multiply SetList.')

    def __imul__(self, other):
        raise ValueError('Cannot multiply SetList.')

    def __rmul__(self, other):
        raise ValueError('Cannot multiply SetList.')

    def __str__(self):
        return super().__str__()
