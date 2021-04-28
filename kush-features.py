import chess
import chess.pgn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

'''
I will return a pandas dataframe

cols: event, site, data, round, white_username, black_username, white_win, black_win 
white_elo, black_elo, white_rating_diff, black_rating_diff, ECO, opening, termination, 
time_control, utc_date, utc_time, 

number of pawn islands: black_pawn_islands, white_pawn_islands
number of isolated single pawns: black_single_pawns, white_single_pawns
pawns protected by another pawn: black_pawn_supported_pawn, white_pawn_supported_pawn

white_king_in_check, white, king_mobility_top3, black_king_in_check, black_king_mobility_top3

white_pieces_moved_upto_now, black_pieces_moved_upto_now

is_checkmate, is_statemate, is_insufficient_material

'''

data_dir = './data/'

cols = ['move_number', 'black_pawn_islands', 'white_pawn_islands', 
'black_single_pawns', 'white_single_pawns', 'white_pieces_moved_upto_now', 'black_pieces_moved_upto_now', 
'white_can_castle', 'black_can_castle', 'white_has_castled', 'black_has_castled'
'white_king_mobility_top3', 'black_king_mobility_top3', 'current_player_in_check',
'is_checkmate', 'is_statemate', 'is_insufficient_material']


# cols = ['event', 'site', 'date', 'round', 'white_username', 'black_username', 
# 'white_win', 'black_win' 'white_elo', 'black_elo', 'white_rating_diff', 
# 'black_rating_diff', 'ECO', 'opening', 'termination', 'time_control', 
# 'utc_date', 'utc_time', 'black_pawn_islands', 'white_pawn_islands', 
# 'black_single_pawns', 'white_single_pawns', 'black_pawn_supported_pawn',
# 'white_pawn_supported_pawn', 'white_king_in_check', 'white', 'king_mobility_top3', 
# 'black_king_in_check', 'black_king_mobility_top3', 'white_pieces_moved_upto_now', 
# 'black_pieces_moved_upto_now', 'is_checkmate', 'is_statemate', 'is_insufficient_material']

main = pd.DataFrame(columns=cols)

def get_num_single_pawns(chesscolor, board):
    '''
    color = True if white, else False
    '''

    sqset = board.pieces(chess.PAWN, chesscolor)
    print(sqset)





def get_king_mobility(chesscolor, board):
    '''
    color = True if white, else False
    '''
    mob = 3
    sqset = board.pieces(chess.KING, chesscolor)
    loc = list(sqset)[0]
    checker = 8 if chesscolor else -8
    dist = [7,8,9]
    if not chesscolor:
        dist = [l*-1 for l in dist]

    if loc+checker < 64 and loc+checker>=0:
        for di in dist:
            next_sq = loc+di
            if board.piece_at(next_sq):
                mob -= 1
    
    return int(mob)

def get_kush_features(game, df_append):
    board = game.board()
    moves = game.mainline_moves()

    has_white_castled = has_black_castled = False
    turn = '-w'
    # assume that every dataframe is initialized with the initial state of a board
    for move in moves:
        ret = pd.DataFrame(columns=cols)
        move_num = board.fullmove_number
        move_current = str(move_num) + turn

        turn = '-b' if board.turn else '-w' # flip them here, since this is for the next move

        movesan = board.san(move)
        print(movesan)
        if movesan in set(['0-0', '0-0-0']):
            if board.turn:
                has_white_castled = True
            else:
                has_black_castled = True
        
        # see whether current player is in check before we move
        ret['current_player_in_check'] = [board.is_check()]
        
        board.push(move)


        get_num_single_pawns(chess.WHITE, board)


        
        ret['white_king_mobility_top3'] = [get_king_mobility(chess.WHITE, board)]
        ret['black_king_mobility_top3'] = [get_king_mobility(chess.BLACK, board)]

        ret['white_can_castle'] = [board.has_castling_rights(chess.WHITE)]
        ret['black_can_castle'] = [board.has_castling_rights(chess.BLACK)]

        ret['white_has_castled'] = [has_white_castled]
        ret['black_has_castled'] = [has_black_castled]

        ret['move_number'] = [move_current]
        ret['is_checkmate'] = [board.is_checkmate()]
        ret['is_stalemate'] = [board.is_stalemate()]
        ret['is_insufficient_material'] = [board.is_insufficient_material()]

        df_append = df_append.append(ret)
        break         

    return df_append

        


if __name__ == '__main__':
    pgn = open(data_dir + '2013-01.pgn')
    game = chess.pgn.read_game(pgn)
    get_kush_features(game, main)

    # while game:
    #     print(game)

    #     tmp = pd.DataFrame(columns=cols)

    #     # tmp['event'] = [game.headers['Event']]
    #     # tmp['site'] = [game.headers['Site']]
    #     # tmp['date'] = [game.headers['Date']]
    #     # tmp['round'] = [game.headers['Round']]
    #     # tmp['white_username'] = [game.headers['White']]
    #     # tmp['black_username'] = [game.headers['Black']]
    #     # res = game.headers['Result']
    #     # white_win = True if res=='1-0' else False
    #     # tmp['white_win'] = [white_win]
    #     # tmp['black_win'] = [not white_win]
    #     # tmp['white_elo'] = [game.headers['WhiteElo']]
    #     # tmp['black_elo'] = [game.headers['BlackElo']]
    #     # tmp['white_rating_diff'] = [game.headers['WhiteRatingDiff']]
    #     # tmp['black_rating_diff'] = [game.headers['BlackRatingDiff']]
    #     # tmp['ECO'] = [game.headers['ECO']]
    #     # tmp['opening'] = [game.headers['Opening']]
    #     # tmp['termination'] = [game.headers['Termination']]
    #     # tmp['time_control'] = [game.headers['TimeControl']]
    #     # tmp['utc_date'] = [game.headers['UTCDate']]
    #     # tmp['utc_time'] = [game.headers['UTCTime']]

    #     board = game.board()
    #     print(board)
    #     moves = game.mainline_moves()
    #     san_moves = get_san_moves(board, moves, 15)
    #     print(' '.join(san_moves))
    #     print(board)

    #     tmp['is_checkmate'] = board.is_checkmate()
    #     tmp['is_stalemate'] = board.is_stalemate()
    #     tmp['is_insufficient_material'] = board.is_insufficient_material()

        




    #     main = main.append(tmp)
    #     # print(tmp)
    #     # print(main.shape)
        
        
        
    #     break #only call this function for 1 game, itll return a df 
    #     game = chess.pgn.read_game(pgn)

    # board = game.board()
