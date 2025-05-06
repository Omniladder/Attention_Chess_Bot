import sys
import math
import os
from typing import List, Optional
import chess
import chess.svg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QScrollArea, QComboBox, QSlider)
from PySide6.QtCore import Qt, Signal, QThread, QMutex
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QFont, QCursor

# Import the enhanced chess model
try:
    from bigger_model import EnhancedChessModel
except ImportError:
    print("Warning: Enhanced chess model not found. AI functionality will be limited.")
    EnhancedChessModel = None


# Node for Monte Carlo Tree Search in chess
class ChessNode:
    def __init__(self, board: chess.Board, move: Optional[chess.Move] = None, parent=None):
        self.board = board.copy()
        self.move = move  # Move that led to this board position
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        self.player = board.turn  # Which player's turn (True for white, False for black)

    def uct_select_child(self, exploration_weight: float = 1.41) -> 'ChessNode':
        """Select a child node using UCT (Upper Confidence Bound for Trees)"""
        log_visits = math.log(self.visits)

        def uct_score(child):
            exploitation = child.wins / child.visits if child.visits > 0 else 0
            exploration = exploration_weight * math.sqrt(log_visits / child.visits) if child.visits > 0 else float(
                'inf')
            return exploitation + exploration

        return max(self.children, key=uct_score)

    def expand(self) -> Optional['ChessNode']:
        """Expand the tree by adding a child node for an untried move"""
        if not self.untried_moves:
            return None

        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)

        child = ChessNode(new_board, move, self)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        """Update node statistics"""
        self.visits += 1
        self.wins += result


# Monte Carlo Tree Search for chess moves
class MCTS:
    def __init__(self, model, iterations: int = 250, exploration_weight: float = 1.41):
        self.model = model
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    def predict_outcome(self, board: chess.Board) -> float:
        """Use the model to predict game outcome from a position"""
        if self.model is None:
            # Fallback evaluation if no model is available
            return self.simple_evaluation(board)

        prediction = self.model.predict(board)
        # Convert to a single score from white's perspective
        # 1.0 is white win, 0.0 is black win, 0.5 is draw
        white_win_prob = prediction[0]
        draw_prob = prediction[1]
        black_win_prob = prediction[2]

        return white_win_prob + 0.5 * draw_prob

    def simple_evaluation(self, board: chess.Board) -> float:
        """Simple material-based evaluation when no model is available"""
        if board.is_checkmate():
            return 0.0 if board.turn == chess.WHITE else 1.0

        # Piece values (pawn=1, knight/bishop=3, rook=5, queen=9)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King has no material value for this calculation
        }

        # Calculate material balance
        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        # Convert to a probability-like value between 0 and 1
        total_material = white_material + black_material
        if total_material == 0:
            return 0.5  # Draw if no material

        # Convert material advantage to a score between 0 and 1
        # A 5-point material advantage is considered decisive
        advantage = (white_material - black_material) / 20
        score = 0.5 + advantage  # Center around 0.5

        # Clamp between 0.05 and 0.95 to avoid absolutes
        return max(0.05, min(0.95, score))

    def simulate(self, board: chess.Board) -> float:
        """Simulate a game from the current position using the evaluation model"""
        # Instead of playing out a random game, use the model to predict the outcome
        outcome = self.predict_outcome(board)

        # Convert to the perspective of the current player
        return outcome if board.turn == chess.WHITE else 1.0 - outcome

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Get the best move according to MCTS"""
        # If there's only one legal move, return it immediately
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]

        root = ChessNode(board)

        for _ in range(self.iterations):
            # Selection
            node = root
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child(self.exploration_weight)

            # Expansion
            if node.untried_moves != []:
                node = node.expand()

            # Simulation
            result = self.simulate(node.board)

            # Backpropagation
            while node is not None:
                node.update(result)
                # Flip result because chess is a zero-sum game
                result = 1.0 - result
                node = node.parent

        # Return the move with the highest visit count
        return max(root.children, key=lambda c: c.visits).move


# Thread for AI chess player to avoid freezing the GUI
class ChessPlayerThread(QThread):
    move_ready = Signal(object)  # Signal emitting a chess.Move

    def __init__(self, model, board: chess.Board, depth: int = 1000):
        super().__init__()
        self.model = model
        self.board = board.copy()
        self.depth = depth
        self.mutex = QMutex()
        self.is_running = True

    def run(self):
        # Create MCTS with the given model
        mcts = MCTS(self.model, iterations=self.depth)

        # Find the best move
        best_move = mcts.get_best_move(self.board)

        # Emit the move
        if self.is_running:  # Check if we've been stopped
            self.move_ready.emit(best_move)

    def stop(self):
        self.mutex.lock()
        self.is_running = False
        self.mutex.unlock()


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
        self.setFixedSize(625, 625)
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
        # Make the move
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
        self.status_label = QLabel("White to move")
        font = QFont()
        font.setPointSize(16)
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
        self.scroll_area.setFixedHeight(400)
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
        elif board.can_claim_threefold_repetition():
            self.status_label.setText("Draw by Threefold Repetition")
        elif board.can_claim_fifty_moves():
            self.status_label.setText("Draw by Fifty-Move Rule")
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
    player_config_changed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create main layout
        main_layout = QVBoxLayout()

        # Create button layout
        button_layout = QHBoxLayout()

        # Control buttons
        self.new_game_btn = (QPushButton("New Game"))
        self.new_game_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.new_game_btn.setStyleSheet("font-size: 14px;")
        self.new_game_btn.setFixedSize(180, 40)

        self.flip_board_btn = (QPushButton("Flip Board"))
        self.flip_board_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.flip_board_btn.setStyleSheet("font-size: 14px;")
        self.flip_board_btn.setFixedSize(180, 40)

        self.undo_move_btn = (QPushButton("Undo Move"))
        self.undo_move_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.undo_move_btn.setStyleSheet("font-size: 14px;")
        self.undo_move_btn.setFixedSize(180, 40)

        button_layout.addWidget(self.new_game_btn)
        button_layout.addWidget(self.undo_move_btn)
        button_layout.addWidget(self.flip_board_btn)

        # Settings layout - Center the player controls
        settings_layout = QHBoxLayout()

        # Create a container widget for player settings
        player_settings = QWidget()
        player_layout = QHBoxLayout(player_settings)

        # White player selection
        white_label = QLabel("White:")
        white_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.white_player = QComboBox()
        self.white_player.addItems(["Human", "AI"])
        self.white_player.setStyleSheet("font-size: 14px; height: 22px;")
        self.white_player.setFixedWidth(100)

        # Black player selection
        black_label = QLabel("Black:")
        black_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.black_player = QComboBox()
        self.black_player.addItems(["Human", "AI"])
        self.black_player.setStyleSheet("font-size: 14px; height: 22px;")
        self.black_player.setFixedWidth(100)

        # Add all to player settings layout
        player_layout.addWidget(white_label)
        player_layout.addWidget(self.white_player)
        player_layout.addSpacing(20)
        player_layout.addWidget(black_label)
        player_layout.addWidget(self.black_player)

        # Center the player settings
        settings_layout.addStretch(1)
        settings_layout.addWidget(player_settings)
        settings_layout.addStretch(1)

        # Add all layouts to main layout
        main_layout.addLayout(button_layout)
        main_layout.addLayout(settings_layout)

        self.setLayout(main_layout)

        # Connect signals
        self.new_game_btn.clicked.connect(self.new_game_clicked)
        self.flip_board_btn.clicked.connect(self.flip_board_clicked)
        self.undo_move_btn.clicked.connect(self.undo_move_clicked)
        self.white_player.currentTextChanged.connect(self._player_selection_changed)
        self.black_player.currentTextChanged.connect(self._player_selection_changed)

    def _player_selection_changed(self):
        """Handle player selection changes"""
        white = self.white_player.currentText()
        black = self.black_player.currentText()
        self.player_config_changed.emit(white, black)


class ChessEvaluationWidget(QWidget):
    """Widget to display chess position evaluation with simplified UI"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize layout
        layout = QVBoxLayout()

        # Evaluation header
        self.eval_label = QLabel("Position Evaluation")
        self.eval_label.setAlignment(Qt.AlignCenter)
        self.eval_label.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
            color: black;
            background-color: #e0e0e0;
            border: 1px solid #aaaaaa;
            border-radius: 4px;
            padding: 8px;
            margin-bottom: 10px;
        """)

        # Probability labels with consistent styling
        self.white_prob = QLabel("White: 33%")
        self.white_prob.setStyleSheet("font-size: 16px; padding: 5px; font-weight: bold; color: #222222;")

        self.draw_prob = QLabel("Draw: 34%")
        # Draw probability is displayed in the same style as others, just a different color
        self.draw_prob.setStyleSheet("font-size: 16px; padding: 5px; font-weight: bold; color: #222222;")

        self.black_prob = QLabel("Black: 33%")
        self.black_prob.setStyleSheet("font-size: 16px; padding: 5px; font-weight: bold; color: #222222;")

        # Create a container for the probability labels
        prob_widget = QWidget()
        prob_widget.setStyleSheet("""
            background-color: #f5f5f5;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 8px;
        """)

        prob_layout = QVBoxLayout(prob_widget)
        prob_layout.addWidget(self.white_prob)
        prob_layout.addWidget(self.draw_prob)
        prob_layout.addWidget(self.black_prob)

        # Add widgets to layout
        layout.addWidget(self.eval_label)
        layout.addWidget(prob_widget)

        self.setLayout(layout)

        # Keep track of evaluation
        self.probabilities = [0.33, 0.34, 0.33]

    def update_evaluation(self, probabilities: List[float]):
        """Update the evaluation display based on model probabilities"""
        self.probabilities = probabilities

        # Update text labels with percentages
        self.white_prob.setText(f"White: {probabilities[0]:.1%}")
        self.draw_prob.setText(f"Draw: {probabilities[1]:.1%}")
        self.black_prob.setText(f"Black: {probabilities[2]:.1%}")


class ChessMainWindow(QMainWindow):
    """Main window for the chess GUI application"""

    def __init__(self):
        super().__init__()

        # Set window title and fixed size
        self.setWindowTitle("Chess with AI")
        self.setFixedSize(1024, 768)

        # Initialize central widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create left panel for board and controls
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

        # Create right panel for evaluation and game info
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Create evaluation widget
        self.eval_widget = ChessEvaluationWidget()

        # Create game info widget
        self.game_info = ChessGameInfo()

        # Add widgets to right layout
        right_layout.addWidget(self.game_info)
        right_layout.addWidget(self.eval_widget)
        right_panel.setLayout(right_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 1)

        # Set the layout to central widget
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initialize player types (human/AI)
        self.white_player_type = "Human"
        self.black_player_type = "Human"

        # AI depth/strength
        self.ai_depth = 250

        # Initialize the model
        self.model = None
        self.load_model()

        # AI player thread
        self.ai_thread = None

        # Connect signals
        self.board_widget.piece_selected.connect(self.handle_square_select)
        self.control_panel.new_game_clicked.connect(self.new_game)
        self.control_panel.flip_board_clicked.connect(self.board_widget.flip_board)
        self.control_panel.undo_move_clicked.connect(self.undo_move)
        self.control_panel.player_config_changed.connect(self.update_player_config)

        # Update displays
        self.game_info.update_status(self.board_widget.board)
        self.update_evaluation()

        # Check if AI should make first move
        self.check_ai_turn()

    def load_model(self):
        """Load the enhanced chess model"""
        if EnhancedChessModel is None:
            print("Enhanced chess model module not available. Using simplified evaluation.")
            self.model = None
            return

        try:
            print("Loading chess model...")
            self.model = EnhancedChessModel(
                model_width=512,
                model_depth=16,
                num_heads=4
            )

            # Check if trained model exists
            model_path = "./models/enhanced_chess_model.pth"
            if os.path.exists(model_path):
                print(f"Loading model weights from {model_path}")
                self.model.load_model(model_path)
            else:
                print("No trained model found. Using untrained model.")

            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Use a dummy model if loading fails
            self.model = None

    def handle_square_select(self, square_info):
        """Handle square selection on the board"""
        square, prev_square = square_info

        # Get current player type for logging purposes
        current_player_type = self.white_player_type if self.board_widget.board.turn else self.black_player_type

        # If a move was made (either by human or AI), update game state
        if prev_square is not None and len(self.board_widget.board.move_stack) > 0:
            # The move has already been made on the board at this point
            # Now we need to update game state and handle the next turn
            self.handle_move_made()
            return

        # The following is only for initial piece selection by a human player
        # This should only apply if no move was made yet (just selecting a piece)
        if current_player_type != "Human":
            return

    def handle_move_made(self):
        """Handle move made on the board"""
        # Update game info
        self.game_info.add_move(self.board_widget.board)
        self.game_info.update_status(self.board_widget.board)

        # Update evaluation
        self.update_evaluation()

        # Check if game is over
        if self.is_game_over():
            return

        # Check if AI needs to make a move
        self.check_ai_turn()

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        board = self.board_widget.board
        return (board.is_checkmate() or
                board.is_stalemate() or
                board.is_insufficient_material() or
                board.can_claim_threefold_repetition() or
                board.can_claim_fifty_moves())

    def update_evaluation(self):
        """Update the evaluation display"""
        if self.model is None:
            # If no model is available, use a simple evaluation function
            self._update_simple_evaluation()
            return

        try:
            # Get evaluation from model
            probabilities = self.model.predict(self.board_widget.board)
            self.eval_widget.update_evaluation(probabilities)
        except Exception as e:
            print(f"Error updating evaluation: {e}")
            self._update_simple_evaluation()

    def _update_simple_evaluation(self):
        """Update evaluation with a simple material-based approach when no model is available"""
        board = self.board_widget.board

        # Simple material evaluation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        total_material = white_material + black_material
        if total_material == 0:
            probabilities = [0.33, 0.34, 0.33]  # Even
        else:
            # Convert material advantage to probabilities
            advantage = (white_material - black_material) / 20  # Normalize
            white_prob = 0.5 + advantage / 2
            black_prob = 0.5 - advantage / 2

            # Adjust for checkmate
            if board.is_checkmate():
                if board.turn == chess.WHITE:  # Black won
                    white_prob, black_prob = 0.05, 0.95
                else:  # White won
                    white_prob, black_prob = 0.95, 0.05

            # Ensure probabilities are in valid range
            white_prob = max(0.05, min(0.95, white_prob))
            black_prob = max(0.05, min(0.95, black_prob))

            # Calculate draw probability as remainder
            draw_prob = 1.0 - white_prob - black_prob

            # Ensure draw probability is at least 0
            if draw_prob < 0:
                # Adjust white and black proportionally
                total = white_prob + black_prob
                white_prob = white_prob / total
                black_prob = black_prob / total
                draw_prob = 0

            probabilities = [white_prob, draw_prob, black_prob]

        self.eval_widget.update_evaluation(probabilities)

    def new_game(self):
        """Start a new game"""
        # Stop AI thread if running
        self.stop_ai_thread()

        # Reset the board
        self.board_widget.reset_board()

        # Reset game info
        self.game_info.reset_history()
        self.game_info.update_status(self.board_widget.board)

        # Update evaluation
        self.update_evaluation()

        # Check if AI should make first move
        self.check_ai_turn()

    def update_player_config(self, white_player: str, black_player: str):
        """Update player configuration"""
        self.white_player_type = white_player
        self.black_player_type = black_player

        # Check if AI needs to make a move
        self.check_ai_turn()

    def update_ai_depth(self, depth: int):
        """Update AI search depth"""
        self.ai_depth = depth

    def undo_move(self):
        """Undo the last move"""
        # Stop AI thread if running
        self.stop_ai_thread()

        board = self.board_widget.board
        if len(board.move_stack) > 0:
            board.pop()

            # If last move was by AI and current player is human, pop another move
            current_player_type = self.white_player_type if board.turn else self.black_player_type
            last_player_type = self.black_player_type if board.turn else self.white_player_type
            if current_player_type == "Human" and last_player_type == "AI" and len(board.move_stack) > 0:
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

            # Update evaluation
            self.update_evaluation()

            # Check if AI needs to make a move
            self.check_ai_turn()

    def check_ai_turn(self):
        """Check if it's AI's turn to move"""
        if self.is_game_over():
            return

        current_player_type = self.white_player_type if self.board_widget.board.turn else self.black_player_type
        if current_player_type == "AI":
            self.make_ai_move()

    def make_ai_move(self):
        """Make an AI move"""
        try:
            # Stop previous AI thread if running
            self.stop_ai_thread()

            # Start a new AI thread
            self.ai_thread = ChessPlayerThread(
                self.model,
                self.board_widget.board.copy(),
                self.ai_depth
            )
            self.ai_thread.move_ready.connect(self.handle_ai_move)
            self.ai_thread.start()
        except Exception as e:
            print(f"Error making AI move: {e}")

    def handle_ai_move(self, move: chess.Move):
        """Handle AI move"""
        # Make the move on the board
        self.board_widget.make_move(move)

        # Update game info
        self.handle_move_made()

    def stop_ai_thread(self):
        """Stop the AI thread if it's running"""
        if self.ai_thread is not None and self.ai_thread.isRunning():
            self.ai_thread.stop()
            self.ai_thread.wait()
            self.ai_thread = None

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop AI thread when window is closed
        self.stop_ai_thread()
        super().closeEvent(event)


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