class DTYPES:
    """A class denoting the list of possible data types."""

    category = "category"
    numeric = "numeric"
    bool = "bool"
    string = "string"
    datetime = "datetime"


class COMPARATORS:
    """A class denoting the list of comparators available for each data type."""

    include = "is in"
    exclude = "is NOT in"
    equals = "equals"
    # list_equals = 'equals (list)'
    not_equal = "does not equal"
    less_than = "less than"
    greater_than = "greater than"
    less_or_equal = "less than or equal"
    greater_or_equal = "greater than or equal"
    contains = "contains"
    not_contain = "does not contain"
    inot_contain = "does not contain (ignore case)"
    startswith = "startswith"
    endswith = "endswith"
    icontains = "contains (ignore case)"
    istartswith = "startswith (ignore case)"
    iendswith = "endswith (ignore case)"
    match = "matches"
    # list_match = 'matches (list)'
    imatch = "matches (ignore case)"
    issame = "is"
    isnot = "is not"
    isbetween = "is between"


CONDITIONOPERATIONS = {
    COMPARATORS.include: "isin",
    COMPARATORS.exclude: "!isin",
    COMPARATORS.issame: "isin",
    COMPARATORS.isnot: "!isin",
    COMPARATORS.equals: "==",
    # COMPARATORS.list_equals: '.isin',
    COMPARATORS.not_equal: "!=",
    COMPARATORS.less_than: "<",
    COMPARATORS.greater_than: ">",
    COMPARATORS.less_or_equal: "<=",
    COMPARATORS.greater_or_equal: ">=",
    COMPARATORS.contains: "contains",
    COMPARATORS.not_contain: "!contains",
    COMPARATORS.inot_contain: "!icontains",
    COMPARATORS.startswith: "startswith",
    COMPARATORS.endswith: "endswith",
    COMPARATORS.match: "match",
    # COMPARATORS.list_match: '.isin',
    COMPARATORS.icontains: "icontains",
    COMPARATORS.istartswith: "istartswith",
    COMPARATORS.iendswith: "iendswith",
    COMPARATORS.imatch: "imatch",
    COMPARATORS.isbetween: "==",  # for date condition, this operation is not actually executed
}


CONDITIONS = {
    DTYPES.category: [
        COMPARATORS.include,
        COMPARATORS.exclude,
        COMPARATORS.issame,
        COMPARATORS.isnot,
    ],
    DTYPES.numeric: [
        COMPARATORS.equals,
        COMPARATORS.less_than,
        COMPARATORS.greater_than,
        COMPARATORS.less_or_equal,
        COMPARATORS.greater_or_equal,
        COMPARATORS.include,
    ],
    DTYPES.bool: [COMPARATORS.equals, COMPARATORS.not_equal],
    DTYPES.string: [
        COMPARATORS.include,
        COMPARATORS.match,
        COMPARATORS.imatch,
        COMPARATORS.contains,
        COMPARATORS.icontains,
        COMPARATORS.startswith,
        COMPARATORS.istartswith,
        COMPARATORS.endswith,
        COMPARATORS.iendswith,
        COMPARATORS.not_contain,
        COMPARATORS.inot_contain,
    ],
    DTYPES.datetime: [
        COMPARATORS.isbetween,
        COMPARATORS.less_than,
        COMPARATORS.greater_than,
        COMPARATORS.less_or_equal,
        COMPARATORS.greater_or_equal,
    ],
}
