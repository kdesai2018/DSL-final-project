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

    :param piece: The piece type.
    :param color: The piece color.
    :param name: The character representing the piece (e.g. 'B' or 'n').
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


class PiecePosition:
    def __init__(self, board: chess.Board, symbol: str, rank: int, file: int, present: int):
        self.symbol = symbol
        self.rank = rank
        self.file = file
        self.position = self.rank * 8 + self.file
        self.present = present

    def __str__(self):
        return f"{self.symbol}: {self.rank}, {self.file}, {self.present}"


class SlidingPiecePosition(PiecePosition):
    def __init__(self, board: chess.Board, *args):
        super().__init__(board, *args)
        self.legal_moves = board.legal_moves
        self.mobility = dict()  # Depends on the piece

    def __str__(self):
        return f"{self.symbol}: {self.rank}, {self.file}, {self.present}, {self.mobility}"

    def get_mobility(self, offsets: Dict[str, int]):
        mobility = {direction: 0 for direction in offsets.keys()}

        for direction, offset in offsets.items():
            dst_idx = self.position + offset
            while dst_idx >= 0 and dst_idx < 64:
                move = chess.Move(self.position, dst_idx)
                if move in self.legal_moves:
                    mobility[direction] += 1
                else:
                    break
                dst_idx += offset

        return mobility

    def diagonal_mobility(self):
        offsets = {'NW': 7, 'NE': 9, 'SW': -9, 'SE': -7}
        return self.get_mobility(offsets)

    def horizontal_mobility(self):
        offsets = {'N': 8, 'S': -8, 'W': -1, 'E': 1}
        return self.get_mobility(offsets)


class KingPosition(PiecePosition):
    def __init__(self, *args):
        super().__init__(*args)


class QueenPosition(SlidingPiecePosition):
    def __init__(self, *args):
        super().__init__(*args)
        self.mobility.update(self.diagonal_mobility())
        self.mobility.update(self.horizontal_mobility())


class RookPosition(SlidingPiecePosition):
    def __init__(self, *args):
        super().__init__(*args)
        self.mobility.update(self.horizontal_mobility())


class BishopPosition(SlidingPiecePosition):
    def __init__(self, *args):
        super().__init__(*args)
        self.mobility.update(self.diagonal_mobility())


class KnightPosition(PiecePosition):
    def __init__(self, *args):
        super().__init__(*args)


class PawnPosition(PiecePosition):
    def __init__(self, *args):
        super().__init__(*args)


PositionClasses = {
    'K': KingPosition,
    'Q': QueenPosition,
    'R': RookPosition,
    'B': BishopPosition,
    'N': KnightPosition,
    'P': PawnPosition,
}


def get_sliding_mobility(board: chess.Board):
    """
    Get the number of spaces sliding pieces can move in each direction.

    :param board: The current board state.
    :returns: TODO (None, currently).
    """
    mobilities = dict()

    for color in Colors:
        for piece in Pieces:
            # Get the string for each piece
            piece_name = chess.Piece(piece, color).symbol()
            cls = PositionClasses[piece_name.upper()]
            coords = get_coordinates(piece, color, piece_name, board)
            for i, position in enumerate(coords):
                key = piece_name + str(i)
                # feat[key] = position
                mobility = cls(board, key, *position)
                mobilities[key] = mobility


def get_general_mobility(board: chess.Board) -> List[int]:
    """
    Get the number of legal moves for each piece.

    :param board: The current board state.
    :returns: An list of the number of moves, ordered by PieceVectorPositions.
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

    return mobility


def get_center_control(board: chess.Board):  # returns a list of 2 elements: black center control and white center control
    allPiecesOnBoard = list()
    wcc = 0  # white center control
    bcc = 0  # black center control

    # Part 1: Number of pieces on the 4 center squares for each player (contributes to bcc and wcc).
    for color in Colors:
        for piece in Pieces:
            # Get the string for each piece
            piece_name = chess.Piece(piece, color).symbol()
            # K, Q, R, B, N, or P for white
            # k, q, r, b, n, or p for black

            cls = PositionClasses[piece_name.upper()]  # calls the corresponding constructor
            # cls = KingPosition, QueenPosition, RookPosition, BishopPosition, KnightPosition, or PawnPosition

            coords = get_coordinates(piece, color, piece_name, board)  # Coordinates: File, rank, present/not present
            for coord in coords:
                if (coord[2] != 0):
                    allPiecesOnBoard.append(coord)

            for i, position in enumerate(coords):  # for each square the given player has a piece of the given type on
                key = piece_name + str(i)  # P0, K1, b2, etc.
                c = cls(board, key, *position)  # piece name and position, and mobility if rook, bishop, or queen
                print(c)
                if (position == (3, 3, 1) or position == (4, 3, 1) or position == (3, 4, 1) or position == (4, 4, 1)):
                    if (color == chess.WHITE):
                        wcc += 1
                    if (color == chess.BLACK):
                        bcc += 1

    # Part 2: Number of pieces that can attack the 4 center squares for each player (contributes to bcc and wcc)
    # print("all pieces on board are", allPiecesOnBoard)
    for color in Colors:
        for piece in Pieces:
            piece_name = chess.Piece(piece, color).symbol()
            cls = PositionClasses[piece_name.upper()]
            coords = get_coordinates(piece, color, piece_name, board)
            for i, position in enumerate(coords):
                if (position[2] == 1):
                    if (piece_name == "P"):
                        if (position == (2, 2, 1) or position == (3, 2, 1) or position == (4, 2, 1) or position == (
                        5, 2, 1) or position == (2, 3, 1) or position == (3, 3, 1) or position == (
                        4, 3, 1) or position == (5, 3, 1)):
                            wcc += 1
                    if (piece_name == "p"):
                        if (position == (2, 5, 1) or position == (3, 5, 1) or position == (4, 5, 1) or position == (
                        5, 5, 1) or position == (2, 4, 1) or position == (3, 4, 1) or position == (
                        4, 4, 1) or position == (5, 4, 1)):
                            bcc += 1
                    if (piece_name == "N"):
                        if (((position[0] == 1 or position[0] == 6) and 2 <= position[1] <= 5) or (
                                (position[0] == 2 or position[0] == 5) and 1 <= position[1] <= 6) or (
                                (position[0] == 3 or position[0] == 4) and (
                                position[1] == 1 or position[1] == 2 or position[1] == 5 or position[1] == 6))):
                            wcc += 1
                    if (piece_name == "n"):
                        if (((position[0] == 1 or position[0] == 6) and 2 <= position[1] <= 5) or (
                                (position[0] == 2 or position[0] == 5) and 1 <= position[1] <= 6) or (
                                (position[0] == 3 or position[0] == 4) and (
                                position[1] == 1 or position[1] == 2 or position[1] == 5 or position[1] == 6))):
                            bcc += 1
                    if (piece_name == "B"):
                        if (position == (2, 2, 1) or position == (2, 3, 1) or position == (2, 4, 1) or position == (
                        2, 5, 1) or position == (5, 2, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (5, 5, 1) or position == (3, 2, 1) or position == (
                        4, 2, 1) or position == (3, 5, 1) or position == (4, 5, 1)):
                            wcc += 1  # if diagonally touching a center square
                        elif ((position == (1, 1, 1) and ((2, 2, 1) not in coords)) or (
                                position == (2, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (1, 2, 1) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 5, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (1, 6, 1) and ((2, 5, 1) not in coords)) or (
                                      position == (2, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (5, 6, 1) and ((4, 5, 1) not in coords)) or (
                                      position == (6, 6, 1) and ((5, 5, 1) not in coords)) or (
                                      position == (6, 5, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (5, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 1, 1) and ((5, 2, 1) not in coords)) or (
                                      position == (6, 2, 1) and ((5, 3, 1) not in coords))):
                            wcc += 1  # one empty space between bishop and a center square
                        elif ((position == (0, 0, 1) and ((1, 1, 1) not in coords) and ((2, 2, 1) not in coords)) or (
                                position == (0, 1, 1) and ((1, 2, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 0, 1) and ((2, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (0, 6, 1) and ((1, 5, 1) not in coords) and (
                                      (2, 4, 1) not in coords)) or (
                                      position == (0, 7, 1) and ((1, 6, 1) not in coords) and (
                                      (2, 5, 1) not in coords)) or (
                                      position == (1, 7, 1) and ((2, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (6, 7, 1) and ((5, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords)) or (
                                      position == (7, 7, 1) and ((6, 6, 1) not in coords) and (
                                      (5, 5, 1) not in coords)) or (
                                      position == (7, 6, 1) and ((6, 5, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (7, 1, 1) and ((6, 2, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 0, 1) and ((6, 1, 1) not in coords) and (
                                      (5, 2, 1) not in coords)) or (
                                      position == (6, 0, 1) and ((5, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords))):
                            wcc += 1  # two empty spaces between bishop and a center square
                    if (piece_name == "b"):
                        if (position == (2, 2, 1) or position == (2, 3, 1) or position == (2, 4, 1) or position == (
                        2, 5, 1) or position == (5, 2, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (5, 5, 1) or position == (3, 2, 1) or position == (
                        4, 2, 1) or position == (3, 5, 1) or position == (4, 5, 1)):
                            bcc += 1  # if diagonally touching a center square
                        elif ((position == (1, 1, 1) and ((2, 2, 1) not in coords)) or (
                                position == (2, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (1, 2, 1) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 5, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (1, 6, 1) and ((2, 5, 1) not in coords)) or (
                                      position == (2, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (5, 6, 1) and ((4, 5, 1) not in coords)) or (
                                      position == (6, 6, 1) and ((5, 5, 1) not in coords)) or (
                                      position == (6, 5, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (5, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 1, 1) and ((5, 2, 1) not in coords)) or (
                                      position == (6, 2, 1) and ((5, 3, 1) not in coords))):
                            bcc += 1  # one empty space between bishop and a center square
                        elif ((position == (0, 0, 1) and ((1, 1, 1) not in coords) and ((2, 2, 1) not in coords)) or (
                                position == (0, 1, 1) and ((1, 2, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 0, 1) and ((2, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (0, 6, 1) and ((1, 5, 1) not in coords) and (
                                      (2, 4, 1) not in coords)) or (
                                      position == (0, 7, 1) and ((1, 6, 1) not in coords) and (
                                      (2, 5, 1) not in coords)) or (
                                      position == (1, 7, 1) and ((2, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (6, 7, 1) and ((5, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords)) or (
                                      position == (7, 7, 1) and ((6, 6, 1) not in coords) and (
                                      (5, 5, 1) not in coords)) or (
                                      position == (7, 6, 1) and ((6, 5, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (7, 1, 1) and ((6, 2, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 0, 1) and ((6, 1, 1) not in coords) and (
                                      (5, 2, 1) not in coords)) or (
                                      position == (6, 0, 1) and ((5, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords))):
                            bcc += 1  # two empty spaces between bishop and a center square
                    if (piece_name == "K"):
                        if (position == (2, 2, 1) or position == (2, 3, 1) or position == (2, 4, 1) or position == (
                        2, 5, 1) or position == (5, 2, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (5, 5, 1) or position == (3, 2, 1) or position == (
                        4, 2, 1) or position == (3, 5, 1) or position == (4, 5, 1)):
                            wcc += 1  # if directly touching a center square
                    if (piece_name == "k"):
                        if (position == (2, 2, 1) or position == (2, 3, 1) or position == (2, 4, 1) or position == (
                        2, 5, 1) or position == (5, 2, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (5, 5, 1) or position == (3, 2, 1) or position == (
                        4, 2, 1) or position == (3, 5, 1) or position == (4, 5, 1)):
                            bcc += 1  # if directly touching a center square
                    if (piece_name == "R"):
                        if (position == (2, 3, 1) or position == (2, 4, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (3, 2, 1) or position == (4, 2, 1) or position == (
                        3, 5, 1) or position == (4, 5, 1)):
                            wcc += 1  # if directly horizontal or vertical to a center square
                        elif ((position == (1, 3, 1) and ((2, 3, 1) not in coords)) or (
                                position == (1, 4, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (4, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 3, 1) and ((5, 3, 1) not in coords)) or (
                                      position == (6, 4, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (3, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (4, 6, 1) and ((4, 5, 1) not in coords))):
                            wcc += 1  # if one empty space between rook and a center square
                        elif ((position == (0, 3, 1) and ((1, 3, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                position == (0, 4, 1) and ((1, 4, 1) not in coords) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 0, 1) and ((3, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (4, 0, 1) and ((4, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords)) or (
                                      position == (7, 3, 1) and ((6, 3, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 4, 1) and ((6, 4, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (3, 7, 1) and ((3, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (4, 7, 1) and ((4, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords))):
                            wcc += 1  # if two empty spaces between rook and a center square
                    if (piece_name == "r"):
                        if (position == (2, 3, 1) or position == (2, 4, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (3, 2, 1) or position == (4, 2, 1) or position == (
                        3, 5, 1) or position == (4, 5, 1)):
                            bcc += 1  # if directly horizontal or vertical to a center square
                        elif ((position == (1, 3, 1) and ((2, 3, 1) not in coords)) or (
                                position == (1, 4, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (4, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 3, 1) and ((5, 3, 1) not in coords)) or (
                                      position == (6, 4, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (3, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (4, 6, 1) and ((4, 5, 1) not in coords))):
                            bcc += 1  # if one empty space between rook and a center square
                        elif ((position == (0, 3, 1) and ((1, 3, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                position == (0, 4, 1) and ((1, 4, 1) not in coords) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 0, 1) and ((3, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (4, 0, 1) and ((4, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords)) or (
                                      position == (7, 3, 1) and ((6, 3, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 4, 1) and ((6, 4, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (3, 7, 1) and ((3, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (4, 7, 1) and ((4, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords))):
                            bcc += 1  # if two empty spaces between rook and a center square
                    if (piece_name == "Q"): #checks rook and bishop attacking positions
                        if (position == (2, 2, 1) or position == (2, 3, 1) or position == (2, 4, 1) or position == (
                        2, 5, 1) or position == (5, 2, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (5, 5, 1) or position == (3, 2, 1) or position == (
                        4, 2, 1) or position == (3, 5, 1) or position == (4, 5, 1)):
                            wcc += 1
                        elif ((position == (1, 1, 1) and ((2, 2, 1) not in coords)) or (
                                position == (2, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (1, 2, 1) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 5, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (1, 6, 1) and ((2, 5, 1) not in coords)) or (
                                      position == (2, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (5, 6, 1) and ((4, 5, 1) not in coords)) or (
                                      position == (6, 6, 1) and ((5, 5, 1) not in coords)) or (
                                      position == (6, 5, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (5, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 1, 1) and ((5, 2, 1) not in coords)) or (
                                      position == (6, 2, 1) and ((5, 3, 1) not in coords))):
                            wcc += 1
                        elif ((position == (0, 0, 1) and ((1, 1, 1) not in coords) and ((2, 2, 1) not in coords)) or (
                                position == (0, 1, 1) and ((1, 2, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 0, 1) and ((2, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (0, 6, 1) and ((1, 5, 1) not in coords) and (
                                      (2, 4, 1) not in coords)) or (
                                      position == (0, 7, 1) and ((1, 6, 1) not in coords) and (
                                      (2, 5, 1) not in coords)) or (
                                      position == (1, 7, 1) and ((2, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (6, 7, 1) and ((5, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords)) or (
                                      position == (7, 7, 1) and ((6, 6, 1) not in coords) and (
                                      (5, 5, 1) not in coords)) or (
                                      position == (7, 6, 1) and ((6, 5, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (7, 1, 1) and ((6, 2, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 0, 1) and ((6, 1, 1) not in coords) and (
                                      (5, 2, 1) not in coords)) or (
                                      position == (6, 0, 1) and ((5, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords))):
                            wcc += 1
                        elif (position == (2, 3, 1) or position == (2, 4, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (3, 2, 1) or position == (4, 2, 1) or position == (
                              3, 5, 1) or position == (4, 5, 1)):
                            wcc += 1
                        elif ((position == (1, 3, 1) and ((2, 3, 1) not in coords)) or (
                                position == (1, 4, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (4, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 3, 1) and ((5, 3, 1) not in coords)) or (
                                      position == (6, 4, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (3, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (4, 6, 1) and ((4, 5, 1) not in coords))):
                            wcc += 1
                        elif ((position == (0, 3, 1) and ((1, 3, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                position == (0, 4, 1) and ((1, 4, 1) not in coords) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 0, 1) and ((3, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (4, 0, 1) and ((4, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords)) or (
                                      position == (7, 3, 1) and ((6, 3, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 4, 1) and ((6, 4, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (3, 7, 1) and ((3, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (4, 7, 1) and ((4, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords))):
                            wcc += 1
                    if (piece_name == "q"): #checks rook and bishop attacking positions
                        if (position == (2, 2, 1) or position == (2, 3, 1) or position == (2, 4, 1) or position == (
                        2, 5, 1) or position == (5, 2, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (5, 5, 1) or position == (3, 2, 1) or position == (
                        4, 2, 1) or position == (3, 5, 1) or position == (4, 5, 1)):
                            bcc += 1
                        elif ((position == (1, 1, 1) and ((2, 2, 1) not in coords)) or (
                                position == (2, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (1, 2, 1) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 5, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (1, 6, 1) and ((2, 5, 1) not in coords)) or (
                                      position == (2, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (5, 6, 1) and ((4, 5, 1) not in coords)) or (
                                      position == (6, 6, 1) and ((5, 5, 1) not in coords)) or (
                                      position == (6, 5, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (5, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 1, 1) and ((5, 2, 1) not in coords)) or (
                                      position == (6, 2, 1) and ((5, 3, 1) not in coords))):
                            bcc += 1
                        elif ((position == (0, 0, 1) and ((1, 1, 1) not in coords) and ((2, 2, 1) not in coords)) or (
                                position == (0, 1, 1) and ((1, 2, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                      position == (1, 0, 1) and ((2, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (0, 6, 1) and ((1, 5, 1) not in coords) and (
                                      (2, 4, 1) not in coords)) or (
                                      position == (0, 7, 1) and ((1, 6, 1) not in coords) and (
                                      (2, 5, 1) not in coords)) or (
                                      position == (1, 7, 1) and ((2, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (6, 7, 1) and ((5, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords)) or (
                                      position == (7, 7, 1) and ((6, 6, 1) not in coords) and (
                                      (5, 5, 1) not in coords)) or (
                                      position == (7, 6, 1) and ((6, 5, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (7, 1, 1) and ((6, 2, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 0, 1) and ((6, 1, 1) not in coords) and (
                                      (5, 2, 1) not in coords)) or (
                                      position == (6, 0, 1) and ((5, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords))):
                            bcc += 1
                        elif (position == (2, 3, 1) or position == (2, 4, 1) or position == (5, 3, 1) or position == (
                        5, 4, 1) or position == (3, 2, 1) or position == (4, 2, 1) or position == (
                              3, 5, 1) or position == (4, 5, 1)):
                            bcc += 1
                        elif ((position == (1, 3, 1) and ((2, 3, 1) not in coords)) or (
                                position == (1, 4, 1) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 1, 1) and ((3, 2, 1) not in coords)) or (
                                      position == (4, 1, 1) and ((4, 2, 1) not in coords)) or (
                                      position == (6, 3, 1) and ((5, 3, 1) not in coords)) or (
                                      position == (6, 4, 1) and ((5, 4, 1) not in coords)) or (
                                      position == (3, 6, 1) and ((3, 5, 1) not in coords)) or (
                                      position == (4, 6, 1) and ((4, 5, 1) not in coords))):
                            bcc += 1
                        elif ((position == (0, 3, 1) and ((1, 3, 1) not in coords) and ((2, 3, 1) not in coords)) or (
                                position == (0, 4, 1) and ((1, 4, 1) not in coords) and ((2, 4, 1) not in coords)) or (
                                      position == (3, 0, 1) and ((3, 1, 1) not in coords) and (
                                      (3, 2, 1) not in coords)) or (
                                      position == (4, 0, 1) and ((4, 1, 1) not in coords) and (
                                      (4, 2, 1) not in coords)) or (
                                      position == (7, 3, 1) and ((6, 3, 1) not in coords) and (
                                      (5, 3, 1) not in coords)) or (
                                      position == (7, 4, 1) and ((6, 4, 1) not in coords) and (
                                      (5, 4, 1) not in coords)) or (
                                      position == (3, 7, 1) and ((3, 6, 1) not in coords) and (
                                      (3, 5, 1) not in coords)) or (
                                      position == (4, 7, 1) and ((4, 6, 1) not in coords) and (
                                      (4, 5, 1) not in coords))):
                            bcc += 1

    center_control_list = list()
    center_control_list.append(bcc)
    center_control_list.append(wcc)
    return center_control_list


def board_to_feat(board: chess.Board):
    """
    Convert a board state into a set of features.

    :param board: The current board state.
    :returns: A dictionary of features.
    """
    feat = dict()

    feat['material'] = get_material(board)
    feat['mobility'] = get_general_mobility(board)
    feat['center_control'] = get_center_control(board)

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
