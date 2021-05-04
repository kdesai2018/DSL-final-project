from features import get_pawn_structure
import chess

board = chess.Board()
board.clear()
pieces = [
    (chess.A4, chess.PAWN, chess.WHITE),
    (chess.B5, chess.PAWN, chess.WHITE),
    (chess.C4, chess.PAWN, chess.WHITE),
    (chess.E2, chess.PAWN, chess.WHITE),
    (chess.G3, chess.PAWN, chess.WHITE),
    (chess.D6, chess.PAWN, chess.BLACK),
    (chess.E6, chess.PAWN, chess.BLACK),
    (chess.F7, chess.PAWN, chess.BLACK),
    (chess.G6, chess.PAWN, chess.BLACK),
    (chess.H4, chess.PAWN, chess.BLACK),
]

for square, piece_type, color in pieces:
    board.set_piece_at(square, chess.Piece(piece_type, color))

display(board)

for color in [chess.WHITE, chess.BLACK]:
    print(get_pawn_structure(color, board))