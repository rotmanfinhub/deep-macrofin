from enum import Enum

class Comparator(str, Enum):
    LEQ = "<="
    GEQ = ">="
    LT = "<"
    GT = ">"
    EQ = "="