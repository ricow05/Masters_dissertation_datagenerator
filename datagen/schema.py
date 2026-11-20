"""Tables with finite, ordered attribute domains, linked as a tree by foreign keys.

Only a tree of depth one (one parent table, one child table) is generated for now. The
schema is still described as a tree, so a third table can be added without changing the
file format.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Table:
    """One table: a key column and attributes with ordered domains.

    ``parent`` names the parent table, and ``foreign_key`` the column of this table that
    refers to the parent's key (by default it has the same name as the parent's key).
    """

    name: str
    key: str
    attributes: dict[str, tuple[str, ...]]
    parent: str | None = None
    foreign_key: str | None = None

    def __post_init__(self):
        self.attributes = {str(a): tuple(str(v) for v in d) for a, d in self.attributes.items()}

    def to_dict(self):
        out = {"name": self.name, "key": self.key, "attributes": {a: list(d) for a, d in self.attributes.items()}}
        if self.parent is not None:
            out["parent"] = self.parent
            out["foreign_key"] = self.foreign_key
        return out

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            key=data["key"],
            attributes={a: tuple(d) for a, d in data["attributes"].items()},
            parent=data.get("parent"),
            foreign_key=data.get("foreign_key"),
        )


@dataclass
class Schema:
    """A root table and one child table.

    Attribute names are unique across tables, so a rule can name attributes without
    saying which table they belong to.
    """

    tables: list[Table]
    domains: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self):
        names = [t.name for t in self.tables]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate table names: {names}")
        roots = [t for t in self.tables if t.parent is None]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root table, found {[t.name for t in roots]}")
        children = [t for t in self.tables if t.parent is not None]
        if len(children) != 1 or len(self.tables) != 2:
            raise NotImplementedError("only one parent table with one child table is supported for now")
        if children[0].parent != roots[0].name:
            raise ValueError(f"{children[0].name}: unknown parent table {children[0].parent!r}")
        if children[0].foreign_key is None:
            children[0].foreign_key = roots[0].key

        reserved = {roots[0].key, children[0].key, children[0].foreign_key}
        if len({roots[0].key, children[0].key}) != 2 or children[0].foreign_key == children[0].key:
            raise ValueError(f"key columns must differ: {sorted(reserved)}")
        self.domains = {}
        for table in self.tables:
            if not table.attributes:
                raise ValueError(f"{table.name}: no attributes")
            for attribute, domain in table.attributes.items():
                if attribute in reserved:
                    raise ValueError(f"{table.name}: attribute {attribute!r} has the name of a key column")
                if attribute in self.domains:
                    raise ValueError(f"attribute {attribute!r} appears in more than one table")
                if len(domain) < 1:
                    raise ValueError(f"{table.name}.{attribute}: empty domain")
                if len(set(domain)) != len(domain):
                    raise ValueError(f"{table.name}.{attribute}: duplicate values in the domain")
                if any(v == "" for v in domain):
                    raise ValueError(f"{table.name}.{attribute}: the empty string is not a valid value")
                self.domains[attribute] = domain

    # -- structure -------------------------------------------------------------------
    @property
    def parent(self) -> Table:
        return next(t for t in self.tables if t.parent is None)

    @property
    def child(self) -> Table:
        return next(t for t in self.tables if t.parent is not None)

    @property
    def parent_attrs(self) -> list[str]:
        return list(self.parent.attributes)

    @property
    def child_attrs(self) -> list[str]:
        return list(self.child.attributes)

    @property
    def attrs(self) -> list[str]:
        return self.parent_attrs + self.child_attrs

    def table_of(self, attribute) -> Table:
        for table in self.tables:
            if attribute in table.attributes:
                return table
        raise KeyError(f"unknown attribute {attribute!r}")

    def is_parent_attr(self, attribute) -> bool:
        return attribute in self.parent.attributes

    def table(self, name) -> Table:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"unknown table {name!r}")

    def index(self, attribute, value) -> int:
        try:
            return self.domains[attribute].index(value)
        except ValueError:
            raise ValueError(f"{value!r} is not in the domain of {attribute}") from None

    def kind(self, rule) -> str:
        """``parent``, ``child`` or ``parent-child``, by the tables the rule's attributes are in."""
        in_parent = [self.is_parent_attr(a) for a in rule.attrs]
        if all(in_parent):
            return "parent"
        if not any(in_parent):
            return "child"
        return "parent-child"

    # -- serialisation ---------------------------------------------------------------
    def to_dict(self):
        return {"tables": [t.to_dict() for t in self.tables]}

    @classmethod
    def from_dict(cls, data):
        return cls([Table.from_dict(t) for t in data["tables"]])


def two_tables(parent_name, parent_key, parent_attributes, child_name, child_key, child_attributes, foreign_key=None):
    """Shorthand for the parent/child schema."""
    return Schema(
        [
            Table(parent_name, parent_key, parent_attributes),
            Table(child_name, child_key, child_attributes, parent=parent_name, foreign_key=foreign_key),
        ]
    )
