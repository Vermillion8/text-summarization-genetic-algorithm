# https://www.geeksforgeeks.org/python-tkinter-text-widget/

import random
import heapq
import nltk
from rouge import Rouge
from nltk.corpus import stopwords as nltk_stopwords
from tkinter import *

nltk.download("punkt")
nltk.download("stopwords")

root = Tk()
root.geometry("800x800")
root.title(" Text Summarizer")

# Objective function
# This function evaluates the fitness of a summary by calculating its ROUGE-L score.
def evaluate_fitness(summary, original_text):
    if summary.strip() == "":
        return 0.0
    else:
        rouge = Rouge()
        scores = rouge.get_scores(summary, original_text)
        return scores[0]["rouge-l"]["f"]


# Initialization
# This function takes a string of text and an integer pop_size as input
# It returns a list of strings, each of which is a randomly generated summary of the original text
# To generate a summary, we first tokenize the original text into sentences
# We then create a list of sentences from the original text
# We then choose half of the sentences at random and join them together to create a summary
# We do this pop_size times to create a list of pop_size summaries
def initialize_population(original_text, pop_size):
    sentences = nltk.sent_tokenize(original_text)
    population = []
    for _ in range(pop_size):
        summary = " ".join(random.sample(sentences, len(sentences) // 2))
        population.append(summary)
    return population


# Selection
# This function takes a population of summaries and the original text and
# returns a list of the top half of summaries in the population. It does this
# by evaluating each summary's fitness score, and then returning the top half
# of the summaries by fitness score.
def selection(population, original_text):
    fitness_scores = [
        (evaluate_fitness(summary, original_text), summary) for summary in population
    ]
    top_summaries = heapq.nlargest(len(population) // 2, fitness_scores)
    return [summary for _, summary in top_summaries]


# Crossover
# This function takes two parents and returns two children. It splits the parents into their constituent sentences, 
# then combines them into children1 and children2. Children1 contains all the sentences in parent1 and parent2, 
# while children2 contains only the sentences that are in both parents.
def crossover(parent1, parent2):
    sentences1 = set(nltk.sent_tokenize(parent1))
    sentences2 = set(nltk.sent_tokenize(parent2))
    child1 = " ".join(sentences1.union(sentences2))
    child2 = " ".join(sentences1.intersection(sentences2))
    return child1, child2


# Mutation
# This function mutates the summary by either adding a sentence or removing a sentence. 
# It first tokenizes the summary into sentences using nltk.sent_tokenize. 
# It then randomly decides whether to remove a sentence or add a sentence. If it decides to remove a sentence, 
# it will remove a random sentence from the list of sentences. If it decides to add a sentence, 
# it will randomly choose a sentence from the list of sentences and add it to a random location in the list of sentences. 
# It then checks if the mutated summary is empty or contains only stopwords. If it is empty or contains only stopwords, 
# it returns the original summary. If it is not empty and does not contain only stopwords, it returns the mutated summary.
def mutate(summary, language):
    sentences = nltk.sent_tokenize(summary)
    if len(sentences) == 0:
        return summary

    if random.random() < 0.5:
        # Remove a sentence
        if len(sentences) > 1:
            sentences.pop(random.randint(0, len(sentences) - 1))
    else:
        # Add a sentence
        sentences.insert(random.randint(0, len(sentences)), random.choice(sentences))

    # Check if the mutated summary is empty or contains only stopwords
    stopwords = nltk.corpus.stopwords.words(language)
    # print(stopwords)
    mutated_summary = " ".join(sentences)
    mutated_summary_words = nltk.word_tokenize(mutated_summary)
    if len(mutated_summary_words) == 0 or all(
        word in stopwords for word in mutated_summary_words
    ):
        return summary

    return mutated_summary


# Genetic algorithm
# This is a genetic algorithm that takes in a text and a language as parameters
# and returns a summary of the text in that language. It uses the text as an
# initial population and culls the population down to the most fit individuals
# over a set number of iterations. It then randomly selects pairs of individuals
# and creates two children from each pair. The children are then mutated and
# added to the population. This process repeats until the population reaches a
# certain size. It then returns the most fit individual from the population.
def genetic_algorithm(original_text, language, pop_size=10, max_iter=50):
    population = initialize_population(original_text, pop_size)
    for _ in range(max_iter):
        selected_summaries = selection(population, original_text)
        new_population = []
        for i in range(len(selected_summaries) - 1):
            child1, child2 = crossover(selected_summaries[i], selected_summaries[i + 1])
            new_population.append(mutate(child1, language))
            new_population.append(mutate(child2, language))
        population = new_population
        if len(population) < 40:
            break

    best_summary = max(population, key=lambda x: evaluate_fitness(x, original_text))
    if not best_summary:  # Check if best_summary is empty
        best_summary = original_text  # Use original text as a fallback
    return best_summary


# Take input from the user and store it in a variable
def takeInput(language):
    # Delete the previous output
    Output.delete("1.0", "end")
    # Get the input from the user
    INPUT = inputtxt.get("1.0", "end-1c")
    # Check the language
    if language == "English":
        # Set the stopwords to the English language
        stopwords = set(nltk_stopwords.words("english"))
    elif language == "Indonesian":
        # Set the stopwords to the Indonesian language
        stopwords = set(nltk_stopwords.words("indonesian"))

    # Remove the stopwords from the input text
    filtered_text = " ".join(
        word for word in nltk.word_tokenize(INPUT) if word.lower() not in stopwords
    )
    # Insert the filtered text into the output box
    Output.insert(END, genetic_algorithm(filtered_text, language))


l = Label(root, text="Enter the text to be summarized")
inputtxt = Text(root, height=20, width=100, bg="light yellow")

Output = Text(root, height=10, width=100, bg="light cyan")

Display1 = Button(
    root, height=5, width=25, text="English", command=lambda: takeInput("English")
)
Display2 = Button(
    root, height=5, width=25, text="Indonesian", command=lambda: takeInput("Indonesian")
)

l.pack()
inputtxt.pack()
Display1.pack()
Display2.pack()
Output.pack()

mainloop()
