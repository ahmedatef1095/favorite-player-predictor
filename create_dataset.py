import pandas as pd

def create_and_save_synthetic_data(output_path="players_dataset.csv"):
    """Creates a synthetic dataset of user preferences and saves it to a CSV file."""
    print(f"Generating synthetic data and saving to {output_path}...")
    data = {
        'club': [
            'Inter Miami', 'FC Barcelona', 'Al-Nassr', 'Real Madrid', 
            'Paris Saint-Germain', 'Manchester City', 'FC Barcelona', 'Al-Nassr', 
            'Manchester City', 'Paris Saint-Germain', 'Liverpool', 'Bayern Munich', 
            'Real Madrid', 'Tottenham Hotspur', 'Manchester United', 'Inter Miami',
            # --- Duplicated entries to allow for stratification ---
            'Paris Saint-Germain', 'Manchester City', 'FC Barcelona', 'Manchester City',
            'Paris Saint-Germain', 'Liverpool', 'Bayern Munich', 'Real Madrid',
            'Tottenham Hotspur', 'Manchester United', 'Inter Miami', 'Al-Nassr',
            'Real Madrid', 'Liverpool', 'Bayern Munich', 'Tottenham Hotspur',
            'Manchester United', 'Inter Miami', 'FC Barcelona', 'Al-Nassr',
            # --- Add missing data to match other lists ---
            'Inter Miami', 'Al-Nassr', 'Paris Saint-Germain', 'Manchester City', 'FC Barcelona',
            'Manchester City', 'Paris Saint-Germain', 'Liverpool', 'Bayern Munich', 'Real Madrid'

        ],
        'national_team': [
            'Argentina', 'Argentina', 'Portugal', 'Portugal', 
            'France', 'Norway', 'Spain', 'Portugal', 
            'Belgium', 'Brazil', 'Egypt', 'Germany',
            'Brazil', 'South Korea', 'Portugal', 'Uruguay',
            # --- Add duplicate entries to allow for stratification ---
            'France', 'Norway', 'Spain', 'Belgium', 'Brazil', 'Egypt',
            'Germany', 'Brazil', 'South Korea', 'Portugal', 'Uruguay', 'Portugal',
            'Brazil', 'Egypt', 'Germany', 'South Korea', 'Portugal',
            'Argentina', 'Spain', 'Portugal',
            # --- Add more data to ensure test_size >= num_classes ---
            'Argentina', 'Portugal', 'France', 'Norway', 'Spain',
            'Belgium', 'Brazil', 'Egypt', 'Germany', 'Brazil'
        ],
        'age': [
            38, 25, 39, 28, 
            25, 23, 22, 35, 
            30, 31, 32, 35,
            23, 31, 29, 37,
            # --- Add duplicate entries to allow for stratification ---
            26, 24, 21, 31, 32, 31, 36, 24, 32, 30, 38, 36,
            25, 33, 36, 33, 30, 39, 23, 40,
            # --- Add more data to ensure test_size >= num_classes ---
            38, 39, 25, 23, 22,
            30, 31, 32, 35, 23
        ],
        'favorite_player': [
            'Lionel Messi', 'Lionel Messi', 'Cristiano Ronaldo', 'Cristiano Ronaldo', 
            'Kylian Mbappé', 'Erling Haaland', 'Gavi', 'Cristiano Ronaldo', 
            'Kevin De Bruyne', 'Neymar Jr.', 'Mohamed Salah', 'Thomas Müller',
            'Vinícius Júnior', 'Son Heung-min', 'Bruno Fernandes', 'Luis Suárez',
            # --- Add duplicate entries to allow for stratification ---
            'Kylian Mbappé', 'Erling Haaland', 'Gavi', 'Kevin De Bruyne', 'Neymar Jr.',
            'Mohamed Salah', 'Thomas Müller', 'Vinícius Júnior', 'Son Heung-min',
            'Bruno Fernandes', 'Luis Suárez', 'Cristiano Ronaldo',
            'Vinícius Júnior', 'Mohamed Salah', 'Thomas Müller', 'Son Heung-min',
            'Bruno Fernandes', 'Lionel Messi', 'Gavi', 'Cristiano Ronaldo',
            # --- Add more data to ensure test_size >= num_classes ---
            'Lionel Messi', 'Cristiano Ronaldo', 'Kylian Mbappé', 'Erling Haaland', 'Gavi',
            'Kevin De Bruyne', 'Neymar Jr.', 'Mohamed Salah', 'Thomas Müller', 'Vinícius Júnior'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print("Dataset successfully created.")

if __name__ == "__main__":
    create_and_save_synthetic_data()