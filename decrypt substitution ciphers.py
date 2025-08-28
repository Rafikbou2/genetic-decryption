import csv
import collections
import re
import random
import time
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

##############################################################################
# Program parameters

# Encrypted chars in the ciphertext
CHARS = 'ABCDEFGHILMNOPQRSTUVZ'

# Size of the population to use for the genetic algorithm
POPULATION_SIZE = 50

# Size of the population slice of best performing solutions to keep at each
# iteration
TOP_POPULATION = 10

# Number of intervals for which the best score has to be stable before aborting
# the genetic algorithm
STABILITY_INTERVALS = 20

# Number of crossovers to execute for each new child in the genetic algorithm
CROSSOVER_COUNT = 2

# Number of random mutations to introduce for each new child in the genetic
# algorithm
MUTATIONS_COUNT = 1

##############################################################################
# Implementation

def ngram(text, n):
    counter = collections.Counter()
    words = re.findall(r'\b[A-Z]+\b', text.upper())
    for word in words:
        for i in range(len(word) - n + 1):
            counter[word[i:i+n]] += 1
    return counter

def read_ngram_frequencies(filename):
    ngram_freq = {}
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            ngram_freq[row[0]] = int(row[1])
    print(f"Loaded {len(ngram_freq)} n-grams from {filename}")
    return ngram_freq

def decode(ciphertext, key):
    return ''.join(key.get(char, char) for char in ciphertext)

def init_mapping():
    repls = set(CHARS)
    mapping = {}
    for c in CHARS:
        if c in mapping:
            continue
        repl = random.choice(list(repls))
        repls.remove(repl)
        repls.discard(c)
        mapping[c] = repl
        mapping[repl] = c
    return mapping

def update_mapping(mapping, char1, char2):
    """
    Correctly swaps the mappings for two characters.
    This ensures that the reciprocal nature of the substitution cipher key is maintained.
    """
    repl1 = mapping[char1]
    repl2 = mapping[char2]
    
    # Swap the mappings of char1 and char2
    mapping[char1] = repl2
    mapping[char2] = repl1

    # Swap the reverse mappings
    mapping[repl1] = char2
    mapping[repl2] = char1

def score(decoded_text, ngram_freq):
    text_ngram = ngram(decoded_text, len(next(iter(ngram_freq))))
    return sum(occurrences * ngram_freq.get(ngram, 0) for ngram, occurrences in text_ngram.items())

def select(population, ciphertext, ngram_freq):
    scores = [(score(decode(ciphertext, p), ngram_freq), p) for p in population]
    sorted_population = sorted(scores, key=lambda x: x[0], reverse=True)
    selected_population = sorted_population[:TOP_POPULATION]
    return selected_population[0][0], [m for _, m in selected_population]

def generate(population):
    new_population = population[:]
    while len(new_population) < POPULATION_SIZE:
        x, y = random.choice(population), random.choice(population)
        child = x.copy()
        for _ in range(CROSSOVER_COUNT):
            char = random.choice(list(CHARS))
            update_mapping(child, char, y[char])
        for _ in range(MUTATIONS_COUNT):
            char1 = random.choice(list(CHARS))
            char2 = random.choice(list(CHARS))
            # Ensure the two characters are different for a valid swap
            while char1 == char2:
                char2 = random.choice(list(CHARS))
            update_mapping(child, char1, char2)
        new_population.append(child)
    return new_population

###############################################################################
# Decryption routine

def decrypt(ciphertext, ngram_freq):
    population = [init_mapping() for _ in range(POPULATION_SIZE)]
    last_score = 0
    last_score_increase = 0
    iterations = 0

    avg_fitness_list = []
    max_fitness_list = []

    while last_score_increase < STABILITY_INTERVALS:
        population = generate(population)
        best_score, population = select(population, ciphertext, ngram_freq)
        
        scores = [score(decode(ciphertext, p), ngram_freq) for p in population]
        average_fitness = sum(scores) / len(scores)
        max_fitness = max(scores)
        best_key = population[0]

        avg_fitness_list.append(average_fitness)
        max_fitness_list.append(max_fitness)

        print(f'[Generation {iterations}]')
        print(f'Average Fitness: {average_fitness}')
        print(f'Max Fitness: {max_fitness}')
        print(f'Key: {"".join(best_key[char] for char in CHARS)}')

        if best_score > last_score:
            last_score_increase = 0
            last_score = best_score
        else:
            last_score_increase += 1
        iterations += 1

    print('Best solution found after {} generations:'.format(iterations))
    best_text = decode(ciphertext, population[0])
    print(best_text)
    print('Key:', ''.join(population[0][char] for char in CHARS))

    return avg_fitness_list, max_fitness_list, best_text

def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
    messagebox.showinfo("File Saved", f"Decrypted text saved to {filename}")

def load_ciphertext():
    global ciphertext_filename
    ciphertext_filename = filedialog.askopenfilename(title="Select Ciphertext File", filetypes=[("Text Files", "*.txt")])
    if ciphertext_filename:
        messagebox.showinfo("Ciphertext File Loaded", "Ciphertext file loaded successfully.")
        update_decrypt_button_state()

def load_trigram():
    global tri_filename
    tri_filename = filedialog.askopenfilename(title="Select Tri-gram Frequency File", filetypes=[("CSV Files", "*.csv")])
    if tri_filename:
        messagebox.showinfo("Tri-gram Frequency File Loaded", "Tri-gram frequency file loaded successfully.")
        update_decrypt_button_state()

def load_bigram():
    global bi_filename
    bi_filename = filedialog.askopenfilename(title="Select Bi-gram Frequency File", filetypes=[("CSV Files", "*.csv")])
    if bi_filename:
        messagebox.showinfo("Bi-gram Frequency File Loaded", "Bi-gram frequency file loaded successfully.")
        update_decrypt_button_state()

def update_decrypt_button_state():
    if ciphertext_filename and tri_filename and bi_filename:
        decrypt_button.config(state=NORMAL)
    else:
        decrypt_button.config(state=DISABLED)

def decrypt_and_plot():
    global tri_fitness_avg, tri_fitness_max, bi_fitness_avg, bi_fitness_max

    try:
        tri_ngram_frequency = read_ngram_frequencies(tri_filename)
        bi_ngram_frequency = read_ngram_frequencies(bi_filename)
        with open(ciphertext_filename, 'r', encoding='utf-8') as fh:
            ciphertext = fh.read().upper()
    except (FileNotFoundError, IndexError) as e:
        messagebox.showerror("Error", f"A required file was not found or is empty: {e}")
        return

    print("Decrypting with Tri-gram frequencies...")
    start_time = time.time()
    tri_fitness_avg, tri_fitness_max, tri_best_text = decrypt(ciphertext, tri_ngram_frequency)
    end_time = time.time()
    tri_time = end_time - start_time

    print("Decrypting with Bi-gram frequencies...")
    start_time = time.time()
    bi_fitness_avg, bi_fitness_max, bi_best_text = decrypt(ciphertext, bi_ngram_frequency)
    end_time = time.time()
    bi_time = end_time - start_time

    print(f"Tri-gram execution time: {tri_time} seconds")
    print(f"Bi-gram execution time: {bi_time} seconds")

    plot_results(tri_fitness_avg, tri_fitness_max, bi_fitness_avg, bi_fitness_max)

    save_text_to_file(tri_best_text, "tri_decrypted_text.txt")
    save_text_to_file(bi_best_text, "bi_decrypted_text.txt")

def plot_results(tri_fitness_avg, tri_fitness_max, bi_fitness_avg, bi_fitness_max):
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121)
    ax1.plot(tri_fitness_avg, label='Tri-gram Average Fitness')
    ax1.plot(bi_fitness_avg, label='Bi-gram Average Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Average Fitness')
    ax1.set_title('Comparison of Average Fitness over Generations')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(tri_fitness_max, label='Tri-gram Max Fitness')
    ax2.plot(bi_fitness_max, label='Bi-gram Max Fitness')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Max Fitness')
    ax2.set_title('Comparison of Max Fitness over Generations')
    ax2.legend()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()

##############################################################################
# GUI Implementation

root = Tk()
root.title("Decryption and Frequency Analysis")
root.geometry("1200x800")

ciphertext_filename = None
tri_filename = None
bi_filename = None

frame = Frame(root)
frame.pack(pady=20)

label = Label(frame, text="Genetic Algorithm for Decrypting Substitution Cipher", font=("Helvetica", 18))
label.pack()

button_frame = Frame(root)
button_frame.pack(pady=20)

load_ciphertext_button = Button(button_frame, text="Load Ciphertext", command=load_ciphertext, padx=20, pady=10, font=("Helvetica", 12))
load_ciphertext_button.pack(side=LEFT, padx=10) # type: ignore

load_trigram_button = Button(button_frame, text="Load Tri-gram Frequency", command=load_trigram, padx=20, pady=10, font=("Helvetica", 12))
load_trigram_button.pack(side=LEFT, padx=10) # type: ignore

load_bigram_button = Button(button_frame, text="Load Bi-gram Frequency", command=load_bigram, padx=20, pady=10, font=("Helvetica", 12))
load_bigram_button.pack(side=LEFT, padx=10) # type: ignore

decrypt_button = Button(button_frame, text="Decrypt and Plot", command=decrypt_and_plot, state=DISABLED, padx=20, pady=10, font=("Helvetica", 12)) # type: ignore
decrypt_button.pack(side=LEFT, padx=10) # type: ignore

root.mainloop()
