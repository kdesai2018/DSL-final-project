import chess

import chess.svg

from IPython.display import SVG

load = input("Please input starting FEN (leave blank for new game)\n")
if load == "":
	board = chess.Board()
else:
	board = chess.Board(load)


mode = input("Playing vs AI? y/n\n")


print(board.fen())
print(board)
while True:
	print("Turn", board.fullmove_number)
	if mode == "y":
		move = calc_best_move(board.fen())
		chess.Move.from_uci(move)
	else:
		move = input("Move in san\n")
		# while move not in board.legal_moves:
		# 	move = input("Not a legal move, try again")
		while True:
			try:
				board.push_san(move)
			except ValueError:
				print(board.pseudo_legal_moves)
				move = input("That was not a legal move. Here are your legal moves. Try again\n")
				continue
			else:
				break
	if board.is_game_over():
		break
	if board.is_check():
		print("Black, you are in Check!")
	print(board.fen())
	print(board)
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
	if board.is_game_over():
		break
	if board.is_check():
		print("Black, you are in Check!")
	print(board.fen())
	print(board)


print(chess.Outcome.result())