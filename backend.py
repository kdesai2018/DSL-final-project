import chess

import chess.svg

from IPython.display import SVG

mode = input("Playing vs AI? y/n\n")
if mode == "y":
	mode = True
else:
	mode = False

board = chess.Board()
print(board.fen())
print(board)
while True:
	if mode:
		move = calc_best_move(board.fen())
		chess.Move.from_uci(move)
	else:
		print(board.legal_moves)
		move = input("Move in san\n")
		# while move not in board.legal_moves:
		# 	move = input("Not a legal move, try again")
		while True:
			try:
				board.push_san(move)
			except ValueError:
				print(board.legal_moves)
				move = input("That was not a legal move. Here are your legal moves. Try again\n")
				continue
			else:
				break
	print(board.fen())
	print(board)
	print(board.legal_moves)
	move = input("Move in san\n")
	while True:
			try:
				board.push_san(move)
			except ValueError:
				print(board.legal_moves)
				move = input("That was not a legal move. Here are your legal moves. Try again\n")
				continue
			else:
				break
	print(board.fen())
	print(board)