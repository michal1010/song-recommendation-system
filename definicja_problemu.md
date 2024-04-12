# Definicja problemu biznsesowego

Chcemy stworzyć model, który będzie tworzył playlisty z pasującymi do siebie utworami.

1. Tworzenie playslit na podstawie najpopularniejszych piosenek z danego gatunku pasujące do siebie współczynnikami takimi jak dancebility, energy, loudness, speechiness itp
2. Tworzenie playlist z w danej kategorii. Dalej piosenki powinny być wybierane ze względu na popularnosć i datę wydania.
   1. piosenki do tańczenia - duże dancebility i acousticness
   2. piosenki na imprezę - duże dancebility, energy i liveness
   3. piosenki do relaksu - duże acousticness, intrementality oraz wolne tempo
3. Tworzenie playlist ze względu na datę wydania

## Pytania i uwagi do danych

1. Co oznacza atrybut expolicit? Jest on bardzo niezbalansowany i przy kluczowym znaczeniu mógłby przeszkodzić w klasyfikacji.
2. Co oznacza atrybut valence?


pouplarność vs duration
energy vs dancebility
speachness vs acusticness