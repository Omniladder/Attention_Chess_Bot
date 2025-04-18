import sys
import chess
import chess.svg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QScrollArea)
from PySide6.QtCore import Qt, Signal
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QFont, QCursor


class ChessGuiBoard(QSvgWidget):
    """Chess board widget using SVG for rendering"""
    piece_selected = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.flipped = False
        self.last_move = None
        self.setFixedSize(670, 670)
        self.setMouseTracking(True)
        self.update_board()

    def update_board(self):
        """Update the board SVG"""
        # Handle last move arrow
        arrows = []
        if self.last_move:
            arrows.append(chess.svg.Arrow(self.last_move.from_square, self.last_move.to_square))

        # For selected square
        if self.selected_square is not None:
            # Add a circular arrow on the selected square (different color than move arrows)
            arrows.append(chess.svg.Arrow(
                self.selected_square,
                self.selected_square,
                color="#ffcc00"  # Yellow circular arrow
            ))

        # Get legal move destinations
        legal_moves_squares = set()
        for move in self.legal_moves:
            legal_moves_squares.add(move.to_square)

        # Draw the board with legal moves marked with X
        svg_data = chess.svg.board(
            self.board,
            flipped=self.flipped,
            arrows=arrows,
            squares=legal_moves_squares,  # Only mark legal destinations with X
            size=400
        )

        # Load the final SVG
        self.load(bytearray(svg_data, encoding='utf-8'))

    def mousePressEvent(self, event):
        """Handle mouse click on the board"""
        # Calculate square coordinates from mouse position
        x = event.position().x()
        y = event.position().y()

        # Get board size
        size = min(self.width(), self.height())
        square_size = size / 8

        # Convert to square index (0-63)
        file = int(x / square_size)
        rank = 7 - int(y / square_size) if not self.flipped else int(y / square_size)

        # Adjust for flipped board
        if self.flipped:
            file = 7 - file

        # Convert to chess.Square
        square = chess.square(file, rank)

        # Handle piece selection and moves
        if self.selected_square is None:
            # Check if square has a piece and it's the current player's turn
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                # Get all legal moves from this square
                self.legal_moves = [move for move in self.board.legal_moves if move.from_square == square]
                self.update_board()  # Update to show highlights

        else:
            # Check if the square is a valid destination
            move = None
            for legal_move in self.legal_moves:
                if legal_move.to_square == square:
                    move = legal_move
                    break

            if move:
                # Make the move
                self.board.push(move)
                self.last_move = move
                # Emit signal after the move is made
                self.piece_selected.emit((square, self.selected_square))

            # Clear selection
            self.selected_square = None
            self.legal_moves = []
            self.update_board()

    def mouseMoveEvent(self, event):
        """Handle mouse movement to change cursor when over a piece"""
        # Calculate square coordinates from mouse position
        x = event.position().x()
        y = event.position().y()

        # Get board size
        size = min(self.width(), self.height())
        square_size = size / 8

        # Convert to square index
        file = int(x / square_size)
        rank = 7 - int(y / square_size)

        # Adjust for flipped board
        if self.flipped:
            file = 7 - file
            rank = 7 - rank

        # Check if coordinates are within the board
        if 0 <= file < 8 and 0 <= rank < 8:
            # Convert to chess.Square
            square = chess.square(file, rank)

            # Check if square has a piece and it belongs to the current player
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        # Call parent method to ensure proper event handling
        super().mouseMoveEvent(event)

    def flip_board(self):
        """Flip the board view"""
        self.flipped = not self.flipped
        self.update_board()

    def make_move(self, move: chess.Move):
        """Make a move on the board"""
        self.board.push(move)
        self.last_move = move
        self.selected_square = None
        self.legal_moves = []
        self.update_board()

    def reset_board(self):
        """Reset the board to starting position"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.update_board()


class ChessGameInfo(QWidget):
    """Widget to display game information and controls"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize layout
        layout = QVBoxLayout()

        # Game status
        self.status_label = QLabel("White to Move")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.status_label.setFont(font)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            color: black;
            background-color: #e0e0e0;
            border: 1px solid #aaaaaa;
            border-radius: 4px;
            padding: 8px;
            margin-bottom: 10px;
            font-size: 16px;
        """)

        # Move history label
        self.history_label = QLabel("Move History:")
        self.history_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.history_label.setStyleSheet("""
                    padding: 8px;
                    background-color: #555555;
                    color: white;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    border: 1px solid #222222;
                    border-bottom: none;
                    font-size: 16px;
                """)

        # Create a scrollable area for move history
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(550)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #333333;
                border: 1px solid #222222;
                border-top: none;
                border-bottom: none;
            }

            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 14px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background-color: #666666;
                min-height: 20px;
                border-radius: 4px;
                margin: 2px;
                width: 10px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #888888;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        # Create the move history label inside a container widget
        history_container = QWidget()
        history_layout = QVBoxLayout(history_container)
        history_layout.setContentsMargins(0, 0, 0, 0)

        self.move_history = QLabel()
        self.move_history.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.move_history.setStyleSheet("""
            background-color: #333333; 
            color: #ffffff; 
            font-size: 18px;
            padding: 4px 12px;
            font-family: 'Consolas', monospace;
            line-height: 2.0;
            font-weight: normal;
        """)
        history_layout.addWidget(self.move_history)

        # Set the container as the widget for the scroll area
        self.scroll_area.setWidget(history_container)

        # Add widgets to layout
        layout.addWidget(self.status_label)
        layout.addWidget(self.history_label)
        layout.addWidget(self.scroll_area)

        # Move list
        self.move_list = []

        layout.addStretch()
        self.setLayout(layout)

    def update_status(self, board: chess.Board):
        """Update the game status display"""
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            self.status_label.setText(f"Checkmate! {winner} Wins")
        elif board.is_stalemate():
            self.status_label.setText("Stalemate! Game is Drawn")
        elif board.is_insufficient_material():
            self.status_label.setText("Draw by Insufficient Material")
        elif board.is_check():
            turn = "White" if board.turn == chess.WHITE else "Black"
            self.status_label.setText(f"{turn} is in Check")
        else:
            turn = "White" if board.turn == chess.WHITE else "Black"
            self.status_label.setText(f"{turn} to Move")

    def add_move(self, board: chess.Board):
        """Add a move to the history"""
        if len(board.move_stack) == 0:
            return

        # Get the last move
        last_move = board.move_stack[-1]

        # Create a temporary board with the position before the last move
        temp_board = chess.Board()
        for i in range(len(board.move_stack) - 1):
            temp_board.push(board.move_stack[i])

        # Get the SAN representation of the last move
        san = temp_board.san(last_move)

        # Add move number every 2 moves (odd for White, even for Black)
        if len(board.move_stack) % 2 == 1:  # White's move
            move_number = (len(board.move_stack) + 1) // 2
            self.move_list.append(f"{move_number}. {san}")
        else:  # Black's move
            if len(self.move_list) == 0:
                # Handle rare case where game might start with Black's move
                move_number = len(board.move_stack) // 2
                self.move_list.append(f"{move_number}. ... {san}")
            else:
                self.move_list[-1] += f" {san}"

        # Update the move history display
        self.move_history.setText("\n".join(self.move_list))

        # Auto-scroll to the bottom to show the latest move
        v_scroll = self.scroll_area.verticalScrollBar()
        v_scroll.setValue(v_scroll.maximum())

    def reset_history(self):
        """Reset the move history"""
        self.move_list = []
        self.move_history.setText("")


class ChessControlPanel(QWidget):
    """Panel with game control buttons"""
    new_game_clicked = Signal()
    flip_board_clicked = Signal()
    undo_move_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize layout
        layout = QHBoxLayout()

        # Control buttons
        self.new_game_btn = (QPushButton("New Game"))
        self.new_game_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.new_game_btn.setStyleSheet("font-size: 14px;")
        self.new_game_btn.setFixedSize(180,40)
        self.flip_board_btn = (QPushButton("Flip Board"))
        self.flip_board_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.flip_board_btn.setStyleSheet("font-size: 14px;")
        self.flip_board_btn.setFixedSize(180, 40)
        self.undo_move_btn = (QPushButton("Undo Move"))
        self.undo_move_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.undo_move_btn.setStyleSheet("font-size: 14px;")
        self.undo_move_btn.setFixedSize(180, 40)

        layout.addWidget(self.new_game_btn)
        layout.addWidget(self.undo_move_btn)
        layout.addWidget(self.flip_board_btn)

        self.setLayout(layout)

        # Connect signals
        self.new_game_btn.clicked.connect(self.new_game_clicked)
        self.flip_board_btn.clicked.connect(self.flip_board_clicked)
        self.undo_move_btn.clicked.connect(self.undo_move_clicked)


class ChessMainWindow(QMainWindow):
    """Main window for the chess GUI application"""
    def __init__(self):
        super().__init__()

        # Set window title and fixed size
        self.setWindowTitle("Chess")
        self.setFixedSize(1024, 768)

        # Initialize central widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create left panel for board
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Create chess board widget
        self.board_widget = ChessGuiBoard()

        # Create control panel
        self.control_panel = ChessControlPanel()

        # Add widgets to left layout
        left_layout.addWidget(self.board_widget, 0, Qt.AlignCenter)  # Center the board widget
        left_layout.addWidget(self.control_panel)
        left_panel.setLayout(left_layout)

        # Create right panel for game info
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Create game info widget
        self.game_info = ChessGameInfo()

        # Add widgets to right layout
        right_layout.addWidget(self.game_info)
        right_panel.setLayout(right_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 1)

        # Set the layout to central widget
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self.board_widget.piece_selected.connect(self.handle_square_select)
        self.control_panel.new_game_clicked.connect(self.new_game)
        self.control_panel.flip_board_clicked.connect(self.board_widget.flip_board)
        self.control_panel.undo_move_clicked.connect(self.undo_move)

        # Update displays
        self.game_info.update_status(self.board_widget.board)

    def handle_square_select(self, square_info):
        """Handle square selection on the board"""
        square, prev_square = square_info

        # If a move was made, update game state
        if prev_square is not None and len(self.board_widget.board.move_stack) > 0:
            self.handle_move_made()

    def handle_move_made(self):
        """Handle move made on the board"""
        # Update game info
        self.game_info.add_move(self.board_widget.board)
        self.game_info.update_status(self.board_widget.board)

    def new_game(self):
        """Start a new game"""
        # Reset the board
        self.board_widget.reset_board()

        # Reset game info
        self.game_info.reset_history()
        self.game_info.update_status(self.board_widget.board)

    def undo_move(self):
        """Undo the last move"""
        board = self.board_widget.board
        if len(board.move_stack) > 0:
            board.pop()

            # Update the board display
            self.board_widget.selected_square = None
            self.board_widget.legal_moves = []
            self.board_widget.last_move = None if len(board.move_stack) == 0 else board.peek()
            self.board_widget.update_board()

            # Update game info
            self.game_info.reset_history()
            for i in range(len(board.move_stack)):
                move = board.move_stack[i]
                # Get the position before this move
                temp_board = chess.Board()
                for j in range(i):
                    temp_board.push(board.move_stack[j])
                # Add the move in SAN format
                san = temp_board.san(move)
                if i % 2 == 0:
                    move_number = (i // 2) + 1
                    self.game_info.move_list.append(f"{move_number}. {san}")
                else:
                    self.game_info.move_list[-1] += f" {san}"

                # Make the move on the temporary board
                temp_board.push(move)

            self.game_info.move_history.setText("\n".join(self.game_info.move_list))
            self.game_info.update_status(board)


def main():
    """Main function to run the chess GUI"""
    # Create the application
    app = QApplication(sys.argv)

    # Create and show the main window
    main_window = ChessMainWindow()
    main_window.show()

    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()