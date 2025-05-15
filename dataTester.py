#  Author: Kyle Tranfaglia
#  Title: dataTester - data cleaning, filtering, and concatenation
#  Last updated: 05/06/25
#  Description: This program uses the pandas library to store chess games in a data frame after
#  extraction from pgn files plus cleaning, filtering, and concatenation
import pandas as pd
import re
import os

# Read in a pgn into a pandas data frame
def read_pgn(pgn_file_path):
    games = []
    current_headers = {}
    moves = []
    
    with open(pgn_file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = line.strip()
            
            # Empty line separates games
            if not line:
                if moves and 'Result' in current_headers:
                    # Process the moves to remove numbering
                    combined_moves = ' '.join(moves)
                    # Replace move numbers
                    clean_moves = re.sub(r'\d+\.+\s*', '', combined_moves)
                    # Remove result from the move list
                    clean_moves = clean_moves.replace('1-0', '').replace('0-1', '').replace('1/2-1/2', '').replace('*', '')
                    
                    # Only add the game if there are actual moves after cleaning
                    if clean_moves.strip():
                        # Convert result to winner
                        result = current_headers.get('Result', '*')
                        if result == '1-0':
                            winner = 'white'
                        elif result == '0-1':
                            winner = 'black'
                        elif result == '1/2-1/2':
                            winner = 'draw'
                        else:
                            winner = None  # Unknown or ongoing
                            continue
                        
                        # Count the number of moves
                        move_tokens = clean_moves.split()
                        move_count = len(move_tokens)
                        
                        # Skip games with fewer than 30 moves (approx 60 tokens)
                        if move_count < 60:
                            current_headers = {}
                            moves = []
                            continue
                        
                        games.append({
                            'winner': winner,
                            'moves': clean_moves.strip()
                        })
                    
                    current_headers = {}
                    moves = []
                continue
                
            # Grab the headers
            if line.startswith('['):
                header_match = re.match(r'\[(.*?)\s+"(.*?)"\]', line)
                if header_match:
                    header_name, header_value = header_match.groups()
                    current_headers[header_name] = header_value
            # If line doesn't start with '[', it's probably moves
            elif not line.startswith('['):
                moves.append(line)
    
    return pd.DataFrame(games)

# Read all PGN files in a directory
def read_all_pgn_files(data_folder):
    all_games = []
    
    # Get a list of all PGN files in the folder
    pgn_files = [f for f in os.listdir(data_folder) if f.endswith('.pgn')]
    
    # Check if pgn files were found
    if not pgn_files:
        print(f"No PGN files found in {data_folder}")
        return pd.DataFrame()
    
    # Process each file
    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}...")
        pgn_file_path = os.path.join(data_folder, pgn_file)
        
        df = read_pgn(pgn_file_path)
        all_games.append(df)
    
    # Combine all DataFrames
    if all_games:
        combined_df = pd.concat(all_games, ignore_index=True)
        combined_df = combined_df.dropna(subset=['moves'])  # remove any remaining rows with NaN moves
        print(f"\nTotal games loaded: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()

# Balance the dataset by ensuring equal representation of outcomes
def balance_dataset(df):
    # Group by winner
    white_wins = df[df['winner'] == 'white']
    black_wins = df[df['winner'] == 'black']
    draws = df[df['winner'] == 'draw']
    
    # Find the minimum count
    min_count = min(len(white_wins), len(black_wins), len(draws))
    
    # Sample equally from each group
    balanced_white = white_wins.sample(min_count*2, replace=False)
    balanced_black = black_wins.sample(min_count*2, replace=False)
    balanced_draws = draws.sample(min_count, replace=False)
    
    # Combine the balanced datasets
    balanced_df = pd.concat([balanced_white, balanced_black, balanced_draws], ignore_index=True)
    
    return balanced_df

def read_kaggle_chess_data(csv_file_path, max_games=100000, min_elo_diff=0,  min_elo=None):
    # Read the CSV file
    print(f"Reading {csv_file_path}...")
    df = pd.read_csv(csv_file_path, low_memory=False, nrows=max_games)
    
    # Print the original size
    original_size = len(df)
    print(f"Original dataset size: {original_size} games")

    # Apply minimum ELO filter if specified
    if min_elo is not None:
        df = df[(df['WhiteElo'] >= min_elo) & (df['BlackElo'] >= min_elo)]
        print(f"Games after minimum ELO ({min_elo}) filter: {len(df)}")
    
    
    # Filter by ELO difference if specified
    if min_elo_diff > 0:
        # Calculate the absolute ELO difference
        df['elo_diff'] = abs(df['BlackElo'] - df['WhiteElo'])
        # Filter games with sufficient ELO difference
        df = df[df['elo_diff'] >= min_elo_diff]
        print(f"Games after ELO difference filter: {len(df)}")
    
    # Map the result to the winner format
    result_mapping = {
        '1-0': 'white',
        '0-1': 'black',
        '1/2-1/2': 'draw'
    }
    
    # Function to clean move notation
    def clean_moves(moves_text):
        if pd.isna(moves_text):
            return None
            
        # Remove move numbers (e.g., "1.", "2.", etc.)
        cleaned = re.sub(r'\d+\.+\s*', '', moves_text)
        
        # Remove evaluations in curly braces (e.g., "{ [%eval 0.3] }")
        cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
        
        # Remove chess annotation symbols (!, ?, !!, ??, !?, ?!, etc.)
        cleaned = re.sub(r'[!?]+', '', cleaned)
        
        # Remove ellipsis used in move numbering (e.g., "2...")
        cleaned = re.sub(r'\d+\.{3}\s*', '', cleaned)
        
        # Remove final result (e.g., "1-0", "0-1", "1/2-1/2")
        cleaned = re.sub(r'\s+(?:1-0|0-1|1\/2-1\/2|\*)$', '', cleaned)
        
        # Remove any extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    # Create a new DataFrame with just the needed columns
    chess_games = pd.DataFrame({
        'winner': df['Result'].map(result_mapping),
        'moves': df['AN'].apply(clean_moves)
    })
    
    # Drop any rows with NaN values or empty moves
    chess_games = chess_games.dropna()
    chess_games = chess_games[chess_games['moves'].str.strip() != '']
    
    # Filter out games with fewer than 30 moves
    chess_games['move_count'] = chess_games['moves'].apply(lambda x: len(x.split()))
    chess_games = chess_games[chess_games['move_count'] >= 60]
    chess_games = chess_games.drop(columns=['move_count'])
    
    print(f"Final processed dataset size: {len(chess_games)} games")
    return chess_games

# Main Program
if __name__ == "__main__":
    data_folder = "./data"
    kaggle_data_path = "./data/Big_chess_data.csv"

    kaggle_games = read_kaggle_chess_data(
        kaggle_data_path,
        max_games=5000000,  # Set to None to read all games
        min_elo_diff=400,  # Minimum ELO difference between players
        min_elo=1200 # Minimum ELO for both players
    )

    # Read all PGN files and get a combined DataFrame
    all_pgn_games = read_all_pgn_files(data_folder)
    
    # Combine the datasets
    all_chess_games = pd.concat([all_pgn_games, kaggle_games], ignore_index=True)
    
    # Display information about the combined dataset
    if not all_chess_games.empty:
        print(f"\nOriginal dataset:")
        print(f"Total number of games: {all_chess_games.shape[0]}")
        print(f"Results distribution:\n{all_chess_games['winner'].value_counts()}")
        
        # Balance the dataset
        balanced_df = balance_dataset(all_chess_games)
        print(f"Balanced dataset:")
        print(f"Total number of games: {balanced_df.shape[0]}")
        print(f"Results distribution:\n{balanced_df['winner'].value_counts()}")
        
        # Save to CSV
        output_path = "./data/GOATED_data_10.csv"
        balanced_df.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")
    else:
        print("No games were found!")