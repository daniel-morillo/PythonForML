from Logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    Or(AKnight, AKnave), # A is either a Knight or a Knave
    Implication(AKnight, AKnave), # If A is a Knight, then A is a Knave, this is impossible
    Implication(AKnight, And(AKnight, AKnave)) # If A is a Knight, then A is both a Knight and a Knave

)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
   Or(AKnight, AKnave), # A is either a Knight or a Knave
   Or(BKnight, AKnight), # B is either a Knight or a Knave
   Implication(AKnight, And(AKnave, BKnave)), # If A is a Knight, then A and B are both Knaves
   Implication(AKnave, Not(And(AKnave, BKnave))) # If A is a Knave, then A and B are not both Knaves
    
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Or(AKnight, AKnave), # A is either a Knight or a Knave
    Or(BKnight, AKnight), # B is either a Knight or a Knave
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))), # If A is a Knight, then A and B are the same kind
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))), # If A is a Knave, then A and B are not the same kind
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))), # If B is a Knight, then A and B are different kinds
    Implication(BKnave, Or(And(AKnight, BKnave), And(AKnave, BKnight))) # If B is a Knave, then A and B are the same kind
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    Or(AKnight, AKnave), # A is either a Knight or a Knave
    Or(BKnight, BKnave), # B is either a Knight or a Knave
    Or(CKnight, CKnave), # C is either a Knight or a Knave
    Implication(AKnight, Or(AKnight, AKnave)), # If A is a Knight, then A is either a Knight or a Knave
    Implication(AKnave, Not(Or(AKnight, AKnave))), # If A is a Knave, then A is not both a Knight and a Knave
    Implication(BKnight, And(AKnave, CKnave)), # If B is a Knight, then A is a Knave and C is a Knave
    Implication(BKnave, Not(And(AKnave, CKnave))), # If B is a Knave, then A is not a Knave or C is not a Knave
    Implication(CKnight, AKnight), # If C is a Knight, then A is a Knight
    Implication(CKnave, Not(AKnight)) # If C is a Knave, then A is not a Knight
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()