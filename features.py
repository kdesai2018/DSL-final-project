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
    forks = [0, 0, 0, 0]

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

        attackingTheKing = False;
        for sq in list(attackedSquares):
            # print(board.piece_at(x)) #P, Q, None, etc.
            if board.piece_at(sq) is not None:
                attackedColor = chess.Piece.from_symbol(str(board.piece_at(sq))).color
                if (attackedColor == True and attackerColor == False):
                    numBlackAttacks += 1
                    if(board.piece_at(sq) == "K"):
                        attackingTheKing = True
                elif (attackedColor == False and attackerColor == True):
                    numWhiteAttacks += 1
                    if (board.piece_at(sq) == "k"):
                        attackingTheKing = True
        if (numWhiteAttacks > 1):
            forks[1] += 1
            if (attackingTheKing):
                forks[3] += 1
        if (numBlackAttacks > 1):
            forks[0] += 1
            if (attackingTheKing):
                forks[2] += 1

    return forks


def get_pins_and_skewers(board: chess.Board) -> Tuple[int, int, int, int, int, int, int, int]:
    # black's pins, white's pins, black's pins on the king, white's pins on the kings, black's skewers, white's skewers, black's skewers on the king, white's skewers on the king

    # a pin is when you're attacking a piece, and a stronger piece is on the other side of the attacked piece
    # a skewer is when you're attacking a piece, and a weaker piece is on the other side of the attacked piece

    pins_and_skewers = [0, 0, 0, 0, 0, 0, 0, 0]

    # print("finding pins and skewers")
    for square in chess.SQUARES:  # goes from bottom left to bottom right, then upwards
        attackerColor = None
        attackingPieceSymbol = board.piece_at(square)  # R, p, None, etc.
        if (str(attackingPieceSymbol) == "R") or (str(attackingPieceSymbol) == "r") or (
                str(attackingPieceSymbol) == "B") or (str(attackingPieceSymbol) == "b") or (
                str(attackingPieceSymbol) == "Q") or (str(attackingPieceSymbol) == "q"):
            # only bishops, rooks, and queens can possibly skewer or pin a piece
            # print(attackingPieceSymbol, "is attacking")
            attackingPiece = chess.Piece.from_symbol(str(attackingPieceSymbol))  # R, q, B, etc. but as Piece object
            attackerColor = attackingPiece.color  # True for white, False for black, None if no piece on current square

            attackedSquares = board.attacks(
                square)  # list(attackedSquares) = [1, 2, 3, 8, 16] for example (some are empty squares)

            pinnningTheKing = False;  # king is behind the attacked piece
            skeweringTheKing = False;  # king is the attacked piece
            for sq in list(attackedSquares):
                if board.piece_at(sq) is not None:
                    attackedColor = chess.Piece.from_symbol(str(board.piece_at(sq))).color
                    if (attackedColor == True and attackerColor == False):  # if black attacking a white piece
                        # print("black ", attackingPiece, " is attacking white ", board.piece_at(sq))
                        pieceExistsOnOtherSide = False
                        # find the next piece on the other side of the attacked piece
                        directionOfAttack = get_direction_of_attack(square, sq, board)  # square is attacking sq
                        # print(direction_of_attack)
                        # check the next squares in that direction until you find a piece
                        nextSquareInt = sq
                        while True:
                            nextSquareInt = get_next_square(directionOfAttack, nextSquareInt,
                                                            board)  # returns the int for your next square, or -1 if impossible to move that direction
                            if nextSquareInt == -1:
                                break
                            if board.piece_at(nextSquareInt) is not None:
                                pieceExistsOnOtherSide = True
                                break
                        if pieceExistsOnOtherSide:
                            if chess.Piece.from_symbol(str(board.piece_at(
                                    nextSquareInt))).color == True:  # if another white piece is on the other side of the attack
                                # we have a pin or skewer depending on the piece type
                                # print("this is a pin or skewer")
                                attackedPieceIsStronger = compare_pieces(str(board.piece_at(sq)),
                                                                         str(board.piece_at(nextSquareInt)))
                                # print(attackedPieceIsStronger) #false is a pin, true is a skewer
                                if attackedPieceIsStronger:
                                    pins_and_skewers[4] += 1
                                    if str(board.piece_at(sq)) == "K":
                                        pins_and_skewers[6] += 1  # skewer on the king
                                else:
                                    pins_and_skewers[0] += 1
                                    if str(board.piece_at(nextSquareInt)) == "K":
                                        pins_and_skewers[2] += 1  # pin on the king
                    elif (attackedColor == False and attackerColor == True):  # if white attacking a black piece
                        # print("white ", attackingPiece, " is attacking black ", board.piece_at(sq))
                        pieceExistsOnOtherSide = False
                        # find the next piece on the other side of the attacked piece
                        directionOfAttack = get_direction_of_attack(square, sq, board)  # square is attacking sq
                        # check the next squares in that direction until you find a piece
                        nextSquareInt = sq
                        while True:
                            nextSquareInt = get_next_square(directionOfAttack, nextSquareInt,
                                                            board)  # returns the int for your next square, or -1 if impossible to move that direction
                            if nextSquareInt == -1:
                                break
                            if board.piece_at(nextSquareInt) is not None:
                                pieceExistsOnOtherSide = True
                                break
                        if pieceExistsOnOtherSide:
                            if chess.Piece.from_symbol(str(board.piece_at(
                                    nextSquareInt))).color == False:  # if another black piece is on the other side of the attack
                                # we have a pin or skewer depending on the piece type
                                # print("this is a pin or skewer")
                                attackedPieceIsStronger = compare_pieces(str(board.piece_at(sq)),
                                                                         str(board.piece_at(nextSquareInt)))
                                # print(attackedPieceIsStronger) #false is a pin, true is a skewer
                                if attackedPieceIsStronger:
                                    pins_and_skewers[5] += 1
                                    if str(board.piece_at(sq)) == "k":
                                        pins_and_skewers[7] += 1  # skewer on the king
                                else:
                                    pins_and_skewers[1] += 1
                                    if str(board.piece_at(nextSquareInt)) == "k":
                                        pins_and_skewers[3] += 1  # pin on the king

    return pins_and_skewers


# gets direction in which a bishop, rook, or queen is attacking (helper method for get_pins_and_skewers)
def get_direction_of_attack(attackingSquare: chess.Square, attackedSquare: chess.Square, board: chess.Board) -> str:
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
def compare_pieces(attackedPieceString, hiddenPieceString) -> bool:
    if (attackedPieceString == "P") or (attackedPieceString == "p"):
        return False
    if (attackedPieceString == "B") or (attackedPieceString == "b") or (attackedPieceString == "N") or (
            attackedPieceString == "n"):
        if (hiddenPieceString == "P") or (hiddenPieceString == "p"):
            return True
        else:
            return False
    if (attackedPieceString == "R") or (attackedPieceString == "r"):
        if (hiddenPieceString == "P") or (hiddenPieceString == "p") or (hiddenPieceString == "B") or (
                hiddenPieceString == "b") or (hiddenPieceString == "N") or (hiddenPieceString == "n"):
            return True
        else:
            return False
    if (attackedPieceString == "Q") or (attackedPieceString == "q"):
        if (hiddenPieceString == "K") or (hiddenPieceString == "k") or (hiddenPieceString == "Q") or (
                hiddenPieceString == "q"):
            return False
        else:
            return True
    if (attackedPieceString == "K") or (attackedPieceString == "k"):
        return True
    else:
        return True

def get_lowest_piece_controlling_each_square(color: chess.Color,
                                             board: chess.Board):  # True for white, #False for black
    lowest_controller = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    for square in chess.SQUARES:  # 0 to 63
        attackingPieceSymbol = board.piece_at(square)  # R, p, None, etc.
        if attackingPieceSymbol is not None:
            attackingPiece = chess.Piece.from_symbol(str(attackingPieceSymbol))  # R, q, B, etc. but as Piece object
            attackerColor = attackingPiece.color  # True for white, False for black, None if no piece on current square

            if attackerColor == color:  # only check attackers of the passed color
                attackedSquares = board.attacks(
                    square)  # list(attackedSquares) = [1, 2, 3, 8, 16] for example (some are empty squares)
                for sq in list(attackedSquares):
                    if lowest_controller[sq] > piece_to_int(str(attackingPieceSymbol)):
                        lowest_controller[sq] = piece_to_int(str(attackingPieceSymbol))

    # replace any remaining 7's with 0's
    for i in range(64):
        if lowest_controller[i] == 7:
            lowest_controller[i] = 0

    return lowest_controller

def piece_to_int(piece: str):
    if (piece == "P") or (piece == "p"):
        return 1
    if (piece == "N") or (piece == "n"):
        return 2
    if (piece == "B") or (piece == "b"):
        return 3
    if (piece == "R") or (piece == "r"):
        return 4
    if (piece == "Q") or (piece == "q"):
        return 5
    if (piece == "K") or (piece == "k"):
        return 6

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

    print("Board features:", feat)

    print(board)
