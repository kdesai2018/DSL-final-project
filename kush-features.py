import chess
import chess.pgn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

'''
ATTN: trey
The only function you should ideally need to call from this file is `get_kush_features()`

As defined in the function definition, just pass in a game and a df object, and i'll append
the necessary data to the dataframe. There might be a small issue with column names, the `cols` 
list I have below is what I'm using to initialize my dataframe


Apologies in advance for the messy code haha, but everything should work without bugs. 
I'm testing this on random game from 2013-01.pgn and it's run fine, so there shouldn't be an 
syntactical/edge case issues :D


'''



'''
I will return a pandas dataframe

cols: event, site, data, round, white_username, black_username, white_win, black_win 
white_elo, black_elo, white_rating_diff, black_rating_diff, ECO, opening, termination, 
time_control, utc_date, utc_time, 

number of pawn islands: black_pawn_islands, white_pawn_islands
number of isolated single pawns: black_single_pawns, white_single_pawns

white_king_in_check, white, king_mobility_top3, black_king_in_check, black_king_mobility_top3

white_pieces_moved_upto_now, black_pieces_moved_upto_now

is_checkmate, is_statemate, is_insufficient_material

'''

data_dir = './data/'

cols = ['move_number', 'black_pawn_islands', 'white_pawn_islands', 
'black_single_pawns', 'white_single_pawns', 'white_pieces_moved_upto_now', 'black_pieces_moved_upto_now', 
'white_can_castle', 'black_can_castle', 'white_has_castled', 'black_has_castled',
'white_king_mobility_top3', 'black_king_mobility_top3', 'current_player_in_check',
'is_checkmate', 'is_stalemate', 'is_insufficient_material', 'piece_moved']

main = pd.DataFrame(columns=cols)

def get_pawn_structure(chesscolor, board):
    '''
    color = True if white, else False

    an island is defined as two or more pawns next to each other
    return: (single_pawns, islands)
    '''

    sqset = board.pieces(chess.PAWN, chesscolor)


    islands = singletons = 0
    possible_island = False

    for s in range(len(sqset)-1):
        cur_pawn = list(sqset)[s]
        next_pawn = list(sqset)[s+1]

        file_diff = (next_pawn%8) - (cur_pawn%8)
        dist = next_pawn - cur_pawn

        '''
            case 1: single pawn, no pawn in next file -> singleton += 1
            case 2: single pawn with 2 in same file, nothing in next -> singleton+=1
            case 3: single pawn, pawn in next file but not connecting/next to -> island+=1
            case 4: pawn with connecting pawn in next file-> set flag, continue

            here, cases 1 
        '''

        if file_diff != 1: # cases 1 and 2
            singletons += 1
            continue
        elif dist not in set([-7,1,9]):  #case 3
            possible_island = False # resset
            island += 1
            continue
        else: # case 4
            possible_island = True
            continue

    islands += int(possible_island)

    return (singletons, islands)

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

    white_pieces_moved = set()
    black_pieces_moved = set()

    turn = '-w'
    # assume that every dataframe is initialized with the initial state of a board
    for move in moves:
        ret = pd.DataFrame(columns=cols)
        move_num = board.fullmove_number
        move_current = str(move_num) + turn

        turn = '-b' if board.turn else '-w' # flip them here, since this is for the next move

        movesan = board.san(move)

        if movesan in set(['0-0', '0-0-0']):
            if board.turn:
                has_white_castled = True
            else:
                has_black_castled = True
        
        # see whether current player is in check before we move
        ret['current_player_in_check'] = [board.is_check()]
        
        origin = move.from_square

        moved_piece = board.piece_at(origin)
        if moved_piece.color:
            white_pieces_moved.add(moved_piece.piece_type)
        else:
            black_pieces_moved.add(moved_piece.piece_type)

        ret['white_pieces_moved_upto_now'] = [len(white_pieces_moved)]
        ret['black_pieces_moved_upto_now'] = [len(black_pieces_moved)]
        ret['piece_moved'] = moved_piece.piece_type
        board.push(move)



        w_single, w_islands = get_pawn_structure(chess.WHITE, board)
        b_single, b_islands = get_pawn_structure(chess.BLACK, board)

        ret['white_pawn_islands'] = [w_islands]
        ret['black_pawn_islands'] = [b_islands]
        ret['black_single_pawns'] = [b_single]
        ret['white_single_pawns'] = [w_single]

        
        ret['white_king_mobility_top3'] = [get_king_mobility(chess.WHITE, board)]
        ret['black_king_mobility_top3'] = [get_king_mobility(chess.BLACK, board)]

        ret['white_can_castle'] = [board.has_castling_rights(chess.WHITE)]
        ret['black_can_castle'] = [board.has_castling_rights(chess.BLACK)]

        ret['white_has_castled'] = [has_white_castled]
        ret['black_has_castled'] = [has_black_castled]

        ret['move_number'] = [move_current]
        ret['is_checkmate'] = [board.is_checkmate()]
        ret['is_stalemate'] = [board.is_stalemate()]
        print(board.is_stalemate())
        ret['is_insufficient_material'] = [board.is_insufficient_material()]

        df_append = df_append.append(ret)     

    return df_append

        


if __name__ == '__main__':
    pgn = open(data_dir + '2013-01.pgn')
    game = chess.pgn.read_game(pgn)
    game = chess.pgn.read_game(pgn)
    df = get_kush_features(game, main)
    print(game.mainline_moves())
    print(df)
    df.to_csv('tmp.csv')