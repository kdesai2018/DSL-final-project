#!/usr/bin/env python3

# features.py: Extract features from boards

from collections import OrderedDict
from typing import Dict, List, Tuple

import chess
import chess.pgn

data_dir = './data/'

# The initial counts for each piece on the board.
PieceCounts = OrderedDict(
    [
        # White pieces
        ('K', 1),
        ('R', 2),
        ('Q', 1),
        ('B', 2),
        ('N', 2),
        ('P', 8),
        # Black pieces
        ('k', 1),
        ('q', 1),
        ('r', 2),
        ('b', 2),
        ('n', 2),
        ('p', 8),
    ]
)

PieceNames = [piece + str(i) for piece, count in PieceCounts.items() for i in range(count)]

PieceVectorPositions = {piece: i for i, piece in enumerate(PieceNames)}

# This is the most common weight system, but they might not be totally optimal. See this Wikipedia
# page for more info: https://en.wikipedia.org/wiki/Chess_piece_relative_value
PieceWeights = {
    chess.QUEEN: 9,
    chess.ROOK: 5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.PAWN: 1,
}

ColorSigns = {
    chess.WHITE: 1,
    chess.BLACK: -1,
}

Pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

Colors = [chess.WHITE, chess.BLACK]

CenterSquares = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])


def get_material(board: chess.Board) -> (int, int):
    """
    Compute the weighted material for each side.

    :param board: The current board state.
    :returns: A tuple containing the material scores for black and white.
    """
    scores = [0, 0]

    for color in Colors:
        for piece, weight in PieceWeights.items():
            piece_count = len(board.pieces(piece, color))
            scores[color] += piece_count * weight

    return scores


def get_coordinates(
    piece: chess.Piece, color: chess.Color, piece_name: str, board: chess.Board
) -> List[Tuple[int, int, int]]:
    """
    Get the rank and file coordinates of all pieces of a given type/color.

    Only use this function if you need to sort pieces on the board by their coordinates (like
    computing per-piece mobility). There's almost certainly a more elegant way to do things using
    the chess API.

    :param piece: The piece type.
    :param color: The piece color.
    :param piece_name: The character representing the piece (e.g. 'B' or 'n').
    :param board: The current board state.
    :returns: A list of tuples of (rank, file, present_on_board).
    """
    locs = list(board.pieces(piece, color))
    coords = [(int(loc / 8), int(loc % 8), 1) for loc in locs]

    if piece in [chess.PAWN, chess.KNIGHT, chess.ROOK]:
        # Sort by file. This should keep entropy down, I think.
        coords.sort(key=lambda file_rank: file_rank[0])
    elif piece == chess.BISHOP and coords:
        # Keep each bishop in a specific position.
        if len(coords) == 1:
            # We need to have both bishops 'present'
            coords.append((-1, -1, 0))
        if coords[0][0] % 2 == 1:
            coords[0], coords[1] = coords[1], coords[0]

    while len(coords) < PieceCounts[piece_name]:
        coords.append((-1, -1, 0))

    return coords


def get_mobility(board: chess.Board) -> List[int]:
    """
    Get the number of legal moves for each piece.

    :param board: The current board state.
    :returns: A list of the number of moves, ordered by PieceVectorPositions.
    """
    mobility = [0] * len(PieceNames)

    original_turn = board.turn

    # This looks like a nasty loop nest, but the inner loop only runs 32 times.
    for color in Colors:
        board.turn = color
        moves = set(board.legal_moves)
        for piece in Pieces:
            piece_name = chess.Piece(piece, color).symbol()
            coords = get_coordinates(piece, color, piece_name, board)
            for i, coordinates in enumerate(coords):
                key = piece_name + str(i)
                vector_position = PieceVectorPositions[key]
                if not coordinates[2]:
                    # Piece is not in play
                    move_count = -1
                else:
                    # Piece *is* in play. Count how many moves it has.
                    square = coordinates[0] * 8 + coordinates[1]
                    move_count = len([move for move in moves if move.from_square == square])
                mobility[vector_position] = move_count

    board.turn = original_turn

    return mobility


def get_center_control(board: chess.Board) -> Tuple[int, int]:
    """
    Compute the center control for each side.

    :param board: The current board state.
    :returns: A tuple with center control scores for (black, white).
    """
    center_control = [0, 0]

    # How many pieces are on the center?
    for square in CenterSquares:
        color = board.color_at(square)
        if color is not None:
            center_control[color] += 1

    # How many pieces can attack the center?
    for color in Colors:
        attackers = [board.attackers(color, square) for square in CenterSquares]
        attacker_count = sum(map(len, attackers))
        center_control[color] += attacker_count

    return center_control


def get_number_of_forks(board: chess.Board) -> Tuple[int, int]:
    # [number of black pieces that have a fork, number of white pieces that have a fork]
    forks = [0, 0]

    for square in chess.SQUARES:  # goes from bottom left to bottom right, then upwards
        attackerColor = None
        attackingPieceSymbol = board.piece_at(square)
        # print("attacking piece is", attackingPieceSymbol) #R, p, None, etc.
        if attackingPieceSymbol is not None:
            attackingPiece = chess.Piece.from_symbol(str(attackingPieceSymbol))
            # print("attackingPiece is", attackingPiece) ##R, p, None, etc. but as Piece object
            attackerColor = attackingPiece.color

        # print("attackerColor is", attackerColor) #True for white, False for black, None if no piece on current square
        attackedSquares = board.attacks(square)
        # print("attacked squares are")
        # print(list("attackedSquares are", attackedSquares)) #[1, 2, 3, 8, 16] for example are the attackable squares (some are empty)
        numWhiteAttacks = 0
        numBlackAttacks = 0

        for sq in list(attackedSquares):
            # print(board.piece_at(x)) #P, Q, None, etc.
            if board.piece_at(sq) is not None:
                attackedColor = chess.Piece.from_symbol(str(board.piece_at(sq))).color
                if (attackedColor == True and attackerColor == False):
                    numBlackAttacks += 1
                elif (attackedColor == False and attackerColor == True):
                    numWhiteAttacks += 1
        if (numWhiteAttacks > 1):
            forks[1] += 1
        if (numBlackAttacks > 1):
            forks[0] += 1

    return forks

def board_to_feat(board: chess.Board):
    """
    Convert a board state into a set of features.

    :param board: The current board state.
    :returns: A dictionary of features.
    """
    feat = dict()

    feat['material'] = get_material(board)
    feat['mobility'] = get_mobility(board)
    feat['center_control'] = get_center_control(board)
    feat['forks'] = get_number_of_forks(board)

    return feat


def get_san_moves(board: chess.Board, moves: List[chess.Move], move_count: int = 0) -> List[str]:
    """
    Convert a move list to SAN.

    :param board: The starting board state.
    :param moves: The list of moves.
    :param move_count: The maximum number of moves to apply.
    :returns: A list of moves in standard algebraic notation.
    """
    san_moves = list()

    for move in moves:
        san_moves.append(board.san(move))
        board.push(move)  # Apply the move
        move_count -= 1
        if move_count == 0:
            break

    return san_moves


if __name__ == '__main__':
    pgn = open(data_dir + '2013-01.pgn')
    game = chess.pgn.read_game(pgn)

    print(game)

    board = game.board()
    moves = game.mainline_moves()
    san_moves = get_san_moves(board, moves, 15)
    print(' '.join(san_moves))

    feat = board_to_feat(board)

    print("Board features:", feat)

    print(board)
