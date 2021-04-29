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


def get_number_of_forks(board: chess.Board) -> Tuple[int, int, int, int]:
    # [number of black pieces that have a fork, number of white pieces that have a fork,
    # number of black pieces that have a fork on the king, number of white pieces that have a fork on the king]

    # [0] = number of black pieces that have a fork
    # [1] = number of white pieces that have a fork
    # [2] = number of black pieces that have a fork on the king
    # [3] = number of white pieces that have a fork on the king

    ForkIndex = 0
    KingForkIndex = 2

    forks = [0] * 4

    for square in chess.SQUARES:
        attackingPiece = board.piece_at(square)

        if attackingPiece is None:
            continue

        attackerColor = attackingPiece.color

        attackingTheKing = False
        num_attacks = 0
        for sq in board.attacks(square):
            attackedPiece = board.piece_at(sq)
            if attackedPiece is None:
                # No fork on an empty square.
                continue

            if attackedPiece.color == attackerColor:
                # Attacking itself? Not a fork.
                continue

            if board.piece_at(sq) == chess.KING:
                attackingTheKing = True

            num_attacks += 1

        if num_attacks > 1:
            forks[ForkIndex + attackerColor] += 1
            if attackingTheKing:
                forks[KingForkIndex + attackerColor] += 1

    return forks


def get_pins_and_skewers(board: chess.Board) -> Tuple[int, int, int, int, int, int, int, int]:
    # [0] = black's pins
    # [1] = white's pins
    # [2] = black's pins on the king
    # [3] = white's pins on the kings
    # [4] = black's skewers
    # [5] = white's skewers
    # [6] = black's skewers on the king
    # [7] = white's skewers on the king

    # a pin is when you're attacking a piece, and a stronger piece is on the other side of the attacked piece
    # a skewer is when you're attacking a piece, and a weaker piece is on the other side of the attacked piece

    pins_and_skewers = [0] * 8

    # Add color to get the final index.
    PinIdx = 0
    KingPinIdx = 2
    SkewerIdx = 4
    KingSkewerIdx = 6

    # print("finding pins and skewers")
    for square in chess.SQUARES:  # goes from bottom left to bottom right, then upwards
        attackingPiece = board.piece_at(square)

        if attackingPiece is None:
            continue

        if attackingPiece.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
            # only bishops, rooks, and queens can possibly skewer or pin a piece
            continue

        attackerColor = attackingPiece.color
        offset = int(attackerColor)

        # list(attackedSquares) = [1, 2, 3, 8, 16] for example (some are empty squares)
        attackedSquares = board.attacks(square)

        # king is behind the attacked piece
        pinnningTheKing = False

        # king is the attacked piece
        skeweringTheKing = False

        for sq in list(attackedSquares):
            attackedPiece = board.piece_at(sq)

            if attackedPiece is None:
                continue

            attackedColor = attackedPiece.color

            if attackedColor == attackerColor:
                continue

            pieceExistsOnOtherSide = False
            # find the next piece on the other side of the attacked piece
            directionOfAttack = get_direction_of_attack(square, sq, board)  # square is attacking sq
            # print(direction_of_attack)
            # check the next squares in that direction until you find a piece
            nextSquareInt = sq
            while not pieceExistsOnOtherSide:
                # returns the int for your next square, or -1 if impossible to move that direction
                nextSquareInt = get_next_square(directionOfAttack, nextSquareInt, board)
                if nextSquareInt == -1:
                    break
                if board.piece_at(nextSquareInt) is not None:
                    pieceExistsOnOtherSide = True

            if not pieceExistsOnOtherSide:
                continue

            next_piece = board.piece_at(nextSquareInt)

            if next_piece.color != attackedColor:
                continue

            # If another piece of the same color is on the other side of the attack, we
            # have a pin or skewer depending on the piece type.
            attackedPieceIsStronger = compare_pieces(attackedPiece, next_piece)
            # print(attackedPieceIsStronger) #false is a pin, true is a skewer
            if attackedPieceIsStronger:
                pins_and_skewers[SkewerIdx + offset] += 1
                if str(board.piece_at(sq)) == "K":
                    # skewer on the king
                    pins_and_skewers[KingSkewerIdx + offset] += 1
            else:
                pins_and_skewers[PinIdx + offset] += 1
                if str(board.piece_at(nextSquareInt)) == "K":
                    # pin on the king
                    pins_and_skewers[KingPinIdx + offset] += 1

    return pins_and_skewers


# gets direction in which a bishop, rook, or queen is attacking (helper method for get_pins_and_skewers)
def get_direction_of_attack(
    attackingSquare: chess.Square, attackedSquare: chess.Square, board: chess.Board
) -> str:
    direction = ""

    # get coordinates of attacker
    # print("attacker square is", attackingSquare)
    attackerFile = chess.square_file(attackingSquare)
    attackerRank = chess.square_rank(attackingSquare)
    # print("attacker coordinates are", attackerFile, attackerRank) #(x,y) coordinates, 0-indexed

    # get coordinates of attacked piece
    attackedFile = chess.square_file(attackedSquare)
    attackedRank = chess.square_rank(attackedSquare)
    # print("attacked coordinates are", attackedFile, attackedRank)

    if (attackerFile > attackedFile) and (attackerRank < attackedRank):
        direction = "NW"
    elif (attackerFile == attackedFile) and (attackerRank < attackedRank):
        direction = "N"
    elif (attackerFile < attackedFile) and (attackerRank < attackedRank):
        direction = "NE"
    elif (attackerFile < attackedFile) and (attackerRank == attackedRank):
        direction = "E"
    elif (attackerFile < attackedFile) and (attackerRank > attackedRank):
        direction = "SE"
    elif (attackerFile == attackedFile) and (attackerRank > attackedRank):
        direction = "S"
    elif (attackerFile > attackedFile) and (attackerRank > attackedRank):
        direction = "SW"
    elif (attackerFile > attackedFile) and (attackerRank == attackedRank):
        direction = "W"

    return direction


def get_next_square(direction: str, startingSquare: chess.Square, board: chess.Board) -> int:
    # get coordinates of starting square, then move one space in the given direction, then return that new square
    # returns -1 if moving in that direction is not possible
    startingFile = chess.square_file(startingSquare)
    startingRank = chess.square_rank(startingSquare)
    nextFile = -1
    nextRank = -1

    if direction == "NW":
        if (startingFile == 0) or (startingRank == 7):
            return -1
        nextFile = startingFile - 1
        nextRank = startingRank + 1
        return chess.square(nextFile, nextRank)
    elif direction == "N":
        if startingRank == 7:
            return -1
        nextFile = startingFile
        nextRank = startingRank + 1
        return chess.square(nextFile, nextRank)
    elif direction == "NE":
        if (startingFile == 7) or (startingRank == 7):
            return -1
        nextFile = startingFile + 1
        nextRank = startingRank + 1
        return chess.square(nextFile, nextRank)
    elif direction == "E":
        if startingFile == 7:
            return -1
        nextFile = startingFile + 1
        nextRank = startingRank
        return chess.square(nextFile, nextRank)
    elif direction == "SE":
        if (startingFile == 7) or (startingRank == 0):
            return -1
        nextFile = startingFile + 1
        nextRank = startingRank - 1
        return chess.square(nextFile, nextRank)
    elif direction == "S":
        if startingRank == 0:
            return -1
        nextFile = startingFile
        nextRank = startingRank - 1
        return chess.square(nextFile, nextRank)
    elif direction == "SW":
        if (startingFile == 0) or (startingRank == 0):
            return -1
        nextFile = startingFile - 1
        nextRank = startingRank - 1
        return chess.square(nextFile, nextRank)
    else:  # direction is W
        if startingFile == 0:
            return -1
        nextFile = startingFile - 1
        nextRank = startingRank
        return chess.square(nextFile, nextRank)


# return True if attacked piece is stronger (skewer) than hidden piece, false if equal or weaker (pin)
def compare_pieces(attacked_piece, hidden_piece):
    return attacked_piece.piece_type > hidden_piece.piece_type


def get_lowest_piece_controlling_each_square(color: chess.Color, board: chess.Board) -> List[int]:
    NoAttacker = chess.KING + 1
    lowest_controller = [NoAttacker] * len(chess.SQUARES)

    print(len(lowest_controller))

    for square in chess.SQUARES:  # 0 to 63
        attackingPiece = board.piece_at(square)

        if attackingPiece is None:
            continue

        attackerColor = attackingPiece.color

        if attackerColor != color:
            # only check attackers of the passed color
            continue

        attackingType = attackingPiece.piece_type

        # list(attackedSquares) = [1, 2, 3, 8, 16] for example (some are empty squares)
        for sq in board.attacks(square):
            if lowest_controller[sq] > attackingType:
                lowest_controller[sq] = attackingType

    # replace any remaining 7's (NoAttacker) with 0's
    lowest_controller = [0 if cont == NoAttacker else cont for cont in lowest_controller]

    return lowest_controller


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
    feat['pins_and_skewers'] = get_pins_and_skewers(board)
    feat['lowest_black_controllers'] = get_lowest_piece_controlling_each_square(chess.BLACK, board)
    feat['lowest_white_controllers'] = get_lowest_piece_controlling_each_square(chess.WHITE, board)

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

    print("Board features:")
    for name, value in feat.items():
        print(name, value)

    print(board)
