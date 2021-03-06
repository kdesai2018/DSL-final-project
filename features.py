#!/usr/bin/env python3

# features.py: Extract features from boards

import argparse
from collections import OrderedDict
from stockfish import Stockfish
import time
import traceback
from typing import List, Tuple

import pandas as pd

import chess
import chess.pgn
import chess.engine

parser = argparse.ArgumentParser()
parser.add_argument("--game-file", type=str, required=True, help="Game file to parse")
parser.add_argument(
    "--game-count", type=int, required=True, help="Maximum number of games to parse"
)
parser.add_argument("--out-file", type=str, required=True, help="Output file")
parser.add_argument("--skip-games", type=int, default=0, help="Skip first N games")

# Stockfish engine executable path, object, and time limit for analysis.
StockfishModel = None
StockfishModelDepth = 16

# Location of the game file to parse and number of games to consider.
GameFile = None
GameCount = 0
OutputFile = None

# The initial counts for each piece on the board.
PieceCounts = OrderedDict(
    [
        # White pieces
        ('K', 1),
        ('Q', 1),
        ('R', 2),
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

PieceTypes = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

Colors = [chess.WHITE, chess.BLACK]

CenterSquares = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])

ForkTypes = ['black_forks', 'white_forks', 'king_black_forks', 'king_white_forks']

PinSkewerTypes = [
    'black_pins',
    'white_pins',
    'black_king_pins',
    'white_king_pins',
    'black_skewers',
    'white_skewers',
    'black_king_skewers',
    'white_king_skewers',
]


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
    locs = board.pieces(piece, color)
    coords = [(int(loc / 8), int(loc % 8), 1) for loc in locs]

    if piece in [chess.PAWN, chess.KNIGHT, chess.ROOK]:
        # Sort by file. This should keep entropy down, I think.
        coords.sort(key=lambda rank_file: rank_file[1])
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
        for piece_type in PieceTypes:
            piece_name = chess.Piece(piece_type, color).symbol()
            coords = get_coordinates(piece_type, color, piece_name, board)
            for i, coordinates in enumerate(coords):
                key = piece_name + str(i)
                try:
                    vector_position = PieceVectorPositions[key]
                except KeyError:
                    # Promotion
                    continue
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


def get_positions(board: chess.Board) -> List[Tuple[int, int]]:
    """
    Get the location of each piece on the board.

    :param board: The current board state.
    :returns: A list of piece locations, ordered by PieceVectorPositions.
    """
    positions = [0] * len(PieceNames)

    # This looks like a nasty loop nest, but the inner loop only runs 32 times.
    for color in Colors:
        for piece_type in PieceTypes:
            piece_name = chess.Piece(piece_type, color).symbol()
            coords = get_coordinates(piece_type, color, piece_name, board)
            for i, coordinates in enumerate(coords):
                key = piece_name + str(i)
                try:
                    vector_position = PieceVectorPositions[key]
                except KeyError:
                    # Promotion
                    continue
                positions[vector_position] = (coordinates[0], coordinates[1])

    return positions


def get_promotions(board: chess.Board) -> List[Tuple[str, int]]:
    """
    Get the number of extra (promoted) pieces of each type.

    :param board: The current board state.
    :returns: Pairs of (piece symbol, promoted count).
    """
    promotions = list()

    for color in Colors:
        for piece_type in PieceTypes:
            if piece_type == chess.PAWN:
                continue
            piece_name = chess.Piece(piece_type, color).symbol()
            count = len(board.pieces(piece_type, color))
            extras = count - PieceCounts[piece_name]
            extras = extras if extras >= 1 else 0
            key = f"extra_{piece_name}"
            promotions.append((key, extras))

    return promotions


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


def get_pins_and_skewers(
    board: chess.Board,
) -> Tuple[int, int, int, int, int, int, int, int]:
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
            if attackedPieceIsStronger:
                pins_and_skewers[SkewerIdx + offset] += 1
                if str(board.piece_at(sq)) == 'K':
                    # skewer on the king
                    pins_and_skewers[KingSkewerIdx + offset] += 1
            else:
                pins_and_skewers[PinIdx + offset] += 1
                if str(board.piece_at(nextSquareInt)) == 'K':
                    # pin on the king
                    pins_and_skewers[KingPinIdx + offset] += 1

    return pins_and_skewers


# gets direction in which a bishop, rook, or queen is attacking (helper method for get_pins_and_skewers)
def get_direction_of_attack(
    attackingSquare: chess.Square, attackedSquare: chess.Square, board: chess.Board
) -> str:
    direction = ""

    # get coordinates of attacker
    attackerFile = chess.square_file(attackingSquare)
    attackerRank = chess.square_rank(attackingSquare)

    # get coordinates of attacked piece
    attackedFile = chess.square_file(attackedSquare)
    attackedRank = chess.square_rank(attackedSquare)

    if (attackerFile > attackedFile) and (attackerRank < attackedRank):
        direction = 'NW'
    elif (attackerFile == attackedFile) and (attackerRank < attackedRank):
        direction = 'N'
    elif (attackerFile < attackedFile) and (attackerRank < attackedRank):
        direction = 'NE'
    elif (attackerFile < attackedFile) and (attackerRank == attackedRank):
        direction = 'E'
    elif (attackerFile < attackedFile) and (attackerRank > attackedRank):
        direction = 'SE'
    elif (attackerFile == attackedFile) and (attackerRank > attackedRank):
        direction = 'S'
    elif (attackerFile > attackedFile) and (attackerRank > attackedRank):
        direction = 'SW'
    elif (attackerFile > attackedFile) and (attackerRank == attackedRank):
        direction = 'W'

    return direction


def get_next_square(direction: str, startingSquare: chess.Square, board: chess.Board) -> int:
    # get coordinates of starting square, then move one space in the given direction, then return that new square
    # returns -1 if moving in that direction is not possible
    startingFile = chess.square_file(startingSquare)
    startingRank = chess.square_rank(startingSquare)
    nextFile = -1
    nextRank = -1

    if direction == 'NW':
        if (startingFile == 0) or (startingRank == 7):
            return -1
        nextFile = startingFile - 1
        nextRank = startingRank + 1
        return chess.square(nextFile, nextRank)
    elif direction == 'N':
        if startingRank == 7:
            return -1
        nextFile = startingFile
        nextRank = startingRank + 1
        return chess.square(nextFile, nextRank)
    elif direction == 'NE':
        if (startingFile == 7) or (startingRank == 7):
            return -1
        nextFile = startingFile + 1
        nextRank = startingRank + 1
        return chess.square(nextFile, nextRank)
    elif direction == 'E':
        if startingFile == 7:
            return -1
        nextFile = startingFile + 1
        nextRank = startingRank
        return chess.square(nextFile, nextRank)
    elif direction == 'SE':
        if (startingFile == 7) or (startingRank == 0):
            return -1
        nextFile = startingFile + 1
        nextRank = startingRank - 1
        return chess.square(nextFile, nextRank)
    elif direction == 'S':
        if startingRank == 0:
            return -1
        nextFile = startingFile
        nextRank = startingRank - 1
        return chess.square(nextFile, nextRank)
    elif direction == 'SW':
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


def get_square_control(color: chess.Color, board: chess.Board) -> List[int]:
    NoAttacker = chess.KING + 1
    lowest_controller = [NoAttacker] * len(chess.SQUARES)

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


def get_side_controlling_each_square(board: chess.Board):
    # for each piece, if it's white, add 1 to each square it's attacking, otherwise subtract one from each square it's attacking
    controlling_side = [0] * len(chess.SQUARES)

    for square in chess.SQUARES:  # 0 to 63
        attackingPiece = board.piece_at(square)

        if attackingPiece is None:
            continue

        attackerColor = attackingPiece.color
        attackingType = attackingPiece.piece_type

        # list(attackedSquares) = [1, 2, 3, 8, 16] for example (some are empty squares)
        for sq in board.attacks(square):
            if attackerColor == chess.WHITE:
                controlling_side[sq] += 1
            else:
                controlling_side[sq] -= 1

    return controlling_side


def get_pawn_structure(chesscolor, board):
    """
    color = True if white, else False

    an island is defined as two or more pawns next to each other
    return: (single_pawns, islands)
    """

    sqset = list(board.pieces(chess.PAWN, chesscolor))
    sqset = sorted(sqset, key=lambda n: n % 8)
    all_pawns = set(sqset)
    islands = singletons = 0
    possible_island = False

    for s in range(len(sqset) - 1):

        cur_pawn = sqset[s]
        next_pawn = sqset[s + 1]

        file_diff = (next_pawn % 8) - (cur_pawn % 8)
        dist = next_pawn - cur_pawn

        """
            case 1: single pawn, no pawn in next file -> singleton += 1
            case 2: single pawn with 2 in same file, nothing in next -> singleton+=1
            case 3: single pawn, pawn in next file but not connecting/next to -> island+=1
            case 4: pawn with connecting pawn in next file-> set flag, continue

            here, cases 1
        """

        if file_diff > 1:  # cases 1 and 2
            if possible_island:
                islands += int(possible_island)
                possible_island = False
            else:
                singletons += 1
        elif dist not in set([-7, 1, 8, 9]):  # case 3
            possible_island = False  # resset
            islands += 1
            continue
        else:  # case 4
            possible_island = True
            continue

    islands += int(possible_island)

    # check if rightmost of color is a singleton
    right_single_off = [-1, -9, -8]
    try:
        rightmost = sqset[-1]
        right_is_single = True

        for off in right_single_off:
            if rightmost + off >= 0 and rightmost + off <= 63:
                if rightmost + off in all_pawns:
                    right_is_single = False

        singletons += int(right_is_single)
    except IndexError:
        pass

    return (singletons, islands)


def get_king_mobility(chesscolor, board):
    """
    color = True if white, else False
    """
    mob = 3
    sqset = board.pieces(chess.KING, chesscolor)
    loc = list(sqset)[0]
    checker = 8 if chesscolor else -8
    dist = [7, 8, 9]
    if not chesscolor:
        dist = [l * -1 for l in dist]

    if loc + checker < 64 and loc + checker >= 0:
        for di in dist:
            next_sq = loc + di
            try:
                if board.piece_at(next_sq):
                    mob -= 1
            except IndexError:
                pass

    return int(mob)


def get_stockfish_evaluation(board: chess.Board) -> float:
    """
    Evaluate a board using the Stockfish engine.

    :param board: The current board state.
    :returns: Stockfish's score for the board.
    """
    StockfishModel.set_fen_position(board.fen())
    score = StockfishModel.get_evaluation()
    if score['type'] == 'cp':
        return score['value']

    # Checkmate is 30,000 centipawns, and move before that is 1,000 centipawns. So mate in 3
    # evaluates to (roughly) 27,000 centipawns (https://chess.stackexchange.com/q/21862).
    moves_remaining = score['value']
    if moves_remaining < 0:
        # Black to mate
        offset = -30000
    else:
        # White to mate
        offset = 30000
    return offset - 1000 * moves_remaining


def get_features(game):
    board = game.board()
    moves = game.mainline_moves()

    has_white_castled = has_black_castled = False

    white_pieces_moved = set()
    black_pieces_moved = set()

    # Game metadata
    header_dict = dict(game.headers)

    board_states = list()

    for move in moves:
        ret = dict()

        movesan = board.san(move)

        if movesan in set(['0-0', '0-0-0']):
            if board.turn:
                has_white_castled = True
                white_pieces_moved.add(chess.KING)
                white_pieces_moved.add(chess.ROOK)
            else:
                has_black_castled = True
                black_pieces_moved.add(chess.KING)
                black_pieces_moved.add(chess.ROOK)

        # see whether current player is in check before we move
        ret['current_player_in_check'] = int(board.is_check())

        origin = move.from_square

        moved_piece = board.piece_at(origin)
        if moved_piece.color == chess.WHITE:
            white_pieces_moved.add(moved_piece.piece_type)
        else:
            black_pieces_moved.add(moved_piece.piece_type)

        ret['white_pieces_moved_upto_now'] = len(white_pieces_moved)
        ret['black_pieces_moved_upto_now'] = len(black_pieces_moved)
        ret['piece_moved'] = moved_piece.piece_type
        board.push(move)

        w_single, w_islands = get_pawn_structure(chess.WHITE, board)
        b_single, b_islands = get_pawn_structure(chess.BLACK, board)

        ret['white_pawn_islands'] = w_islands
        ret['black_pawn_islands'] = b_islands
        ret['black_single_pawns'] = b_single
        ret['white_single_pawns'] = w_single

        ret['white_king_mobility_top3'] = get_king_mobility(chess.WHITE, board)
        ret['black_king_mobility_top3'] = get_king_mobility(chess.BLACK, board)

        ret['white_can_castle'] = int(board.has_castling_rights(chess.WHITE))
        ret['black_can_castle'] = int(board.has_castling_rights(chess.BLACK))

        ret['white_has_castled'] = int(has_white_castled)
        ret['black_has_castled'] = int(has_black_castled)

        ret['fullmove_number'] = board.fullmove_number
        # It might be easier for a model to use 1 and -1 instead of 1 and 0.
        ret['player_to_move'] = 1 if board.turn == chess.WHITE else -1
        ret['is_checkmate'] = int(board.is_checkmate())
        ret['is_stalemate'] = int(board.is_stalemate())
        ret['is_insufficient_material'] = int(board.is_insufficient_material())

        ret['black_material'], ret['white_material'] = get_material(board)

        for piece, mobility in zip(PieceNames, get_mobility(board)):
            ret[f'{piece}_mobility'] = mobility

        for piece, (rank, file) in zip(PieceNames, get_positions(board)):
            ret[f'{piece}_rank'] = rank
            ret[f'{piece}_file'] = file

        for promoted_piece, count in get_promotions(board):
            ret[promoted_piece] = count

        ret['black_center_control'], ret['white_center_control'] = get_center_control(board)

        for fork_type, fork_count in zip(ForkTypes, get_number_of_forks(board)):
            ret[fork_type] = fork_count

        for pin_skewer_type, pin_skewer_count in zip(PinSkewerTypes, get_pins_and_skewers(board)):
            ret[pin_skewer_type] = pin_skewer_count

        black_control = get_square_control(chess.BLACK, board)
        white_control = get_square_control(chess.WHITE, board)
        for square, black_cont, white_cont in zip(chess.SQUARE_NAMES, black_control, white_control):
            ret[f'black_{square}_control'] = black_cont
            ret[f'white_{square}_control'] = white_cont

        controlling_sides = get_side_controlling_each_square(board)
        for square, control_sum in zip(chess.SQUARE_NAMES, controlling_sides):
            # positive means white has more pieces supporting the square, negative for black
            ret[f'side_controlling_{square}'] = control_sum

        ret['stockfish_evaluation'] = get_stockfish_evaluation(board)

        ret.update(header_dict)

        board_states.append(ret)

        # Uncomment this after adding a new feature to verify the types:
        # for k, v in ret.items():
        #     assert type(k) == str
        #     assert type(v) == int

    return board_states


def get_next_board_features(pgn):
    print('hello')
    game = chess.pgn.read_game(pgn)
    feats = get_features(game)
    print('done')
    return feats


def got_board_features(board_states):
    print("Got {len(board_states)} boards")


if __name__ == '__main__':
    args = parser.parse_args()

    GameFile = args.game_file
    GameCount = args.game_count
    OutputFile = args.out_file

    pgn = open(GameFile)

    StockfishModel = Stockfish(parameters={'Threads': 2})
    StockfishModel.set_depth(StockfishModelDepth)

    print(f"Parsing at most {GameCount} games from {GameFile} into {OutputFile}")

    if args.skip_games:
        for i in range(args.skip_games):
            chess.pgn.read_game(pgn)

    game_count = 0
    records = list()
    try:
        while game_count < GameCount:
            game = chess.pgn.read_game(pgn)
            t0 = time.time()
            board_states = get_features(game)
            elapsed = time.time() - t0
            game_count += 1
            print(f'Processed game {game_count} in {elapsed} seconds ({len(board_states)} boards)')
            records.extend(board_states)
    except Exception as e:
        print(f"Caught exception {e}")
        print(traceback.format_exc())
    finally:
        df = pd.DataFrame.from_records(records, columns=records[0].keys())
        df.to_csv(OutputFile)
