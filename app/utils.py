from models import Position

def manhattan(a: Position, b: Position) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)