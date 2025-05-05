#  Author: Kyle Tranfaglia
#  Title: dataTester - data cleaning, filtering, and concatenation
#  Last updated: 04/07/25
#  Description: This program uses the pandas library to store chess games in a data frame after
#  extraction from pgn files plus cleaning, filtering, and concatenation
import pandas as pd
import re
import os

# Read in a pgn into a pandas data frame
def read_pgn(pgn_file_path):
    games = []
    current_result = None
    # current_eco = None
    moves = []
    
    with open(pgn_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Empty line separates games
            if not line:
                if moves:
                    # Process the moves to remove numbering
                    combined_moves = ' '.join(moves)
                    # Replace move numbers (like "1.", "2.", etc.)
                    clean_moves = re.sub(r'\d+\.+\s*', '', combined_moves)
                    # Remove result from the move list
                    clean_moves = clean_moves.replace('1-0', '').replace('0-1', '').replace('1/2-1/2', '').replace('*', '')

                    
                    games.append({
                        'winner': current_result,
                        # 'ECO': current_eco
                        'moves': clean_moves
                    })
                    current_result = None
                    # current_eco = None  # Reset ECO
                    moves = []
                continue
                
            # Grab the Result header
            if line.startswith('[Result '):
                try:
                    current_result = line.split('"')[1].replace('1-0', 'white').replace('0-1', 'black').replace('1/2-1/2', 'draw').replace('*', 'draw')
                except:
                    pass
            # # Grab the ECO header
            # elif line.startswith('[ECO '):
            #     try:
            #         current_eco = line.split('"')[1]
            #     except:
            #         pass
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
        pgn_file_path = os.path.join(data_folder, pgn_file)
        
        df = read_pgn(pgn_file_path)
        all_games.append(df)
    
    # Combine (concatenate) all DataFrames
    if all_games:
        combined_df = pd.concat(all_games, ignore_index=True)
        print(f"\nTotal games loaded: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()

# Main Program

data_folder = "./data"

# Read all PGN files and get a combined DataFrame
all_chess_games = read_all_pgn_files(data_folder)

# Display information about the combined dataset
if not all_chess_games.empty:
    print(all_chess_games.head())
    print(f"\nTotal number of games: {all_chess_games.shape[0]}")
    print(f"Results distribution:\n{all_chess_games['winner'].value_counts()}")
    
    # Save to CSV
    all_chess_games.to_csv("./data/GM_games_small.csv", index=False)