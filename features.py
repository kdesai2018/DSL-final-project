#!/usr/bin/env python3

# features.py: Extract features from boards

import chess
import chess.pgn

data_dir = './data/'

PieceCounts = {
    # White pieces
    'P': 8,
    'N': 2,
    'B': 2,
    'R': 2,
    'Q': 1,
    'K': 1,
    # Black pieces
    'p': 8,
    'n': 2,
    'b': 2,
    'r': 2,
    'q': 1,
    'k': 1,
}

Pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
Colors = [chess.WHITE, chess.BLACK]


def get_coordinates(piece, color, piece_name, board):
    # Coordinates: File, rank, present/not present
    locs = list(board.pieces(piece, color))
    coords = [(loc % 8, int(loc / 8), 1) for loc in locs]

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
    def __init__(self, game, symbol, rank, file, present):
        self.symbol = symbol
        self.rank = rank
        self.file = file
        self.present = present

    def __str__(self):
        return f"{self.symbol}: {self.rank}, {self.file}, {self.present}"


class SlidingPiecePosition(PiecePosition):
    def __init__(self, board, *args):
        super().__init__(board, *args)
        self.legal_moves = board.legal_moves
        self.mobility = dict()  # Depends on the piece

    def __str__(self):
        return f"{self.symbol}: {self.rank}, {self.file}, {self.present}, {self.mobility}"

    def get_mobility(self, offsets):
        mobility = {direction: 0 for direction in offsets.keys()}

        src_idx = (self.rank - 1) * 8 + (self.file - 1)

        for direction, offset in offsets.items():
            dst_idx = src_idx + offset
            while dst_idx > 0 and dst_idx < 64:
                move = chess.Move(src_idx, dst_idx)
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


def board_to_feat(board):
    feat = dict()

    # 1: Determine the positions of the pieces that are in play.
    for color in Colors:
        for piece in Pieces:
            # Get the string for each piece
            piece_name = chess.Piece(piece, color).symbol()
            cls = PositionClasses[piece_name.upper()]

            coords = get_coordinates(piece, color, piece_name, board)
            for i, position in enumerate(coords):
                key = piece_name + str(i)
                feat[key] = position
                c = cls(board, key, *position)
                print(c)

    # Alternative way to determine piece locations
    # for file in range(8):
    #    for rank in range(8):
    #        square = file * 8 + rank
    #        piece = board.piece_at(square)
    #        if piece:
    #            print(f"{piece} is at coordinate ({rank}, {file})")

    return feat


def get_san_moves(board, moves, move_count=0):
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

    print(board)
