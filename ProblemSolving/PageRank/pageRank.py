import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py directory")
    corpus = crawl(sys.argv[1])
    print(corpus)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    prob_dist = dict()
    #counting links in our page
    linked_pages = corpus[page]
    num_links = len(linked_pages)

    #if there are no links in our page, we are threating it as if it has links to all pages
    if num_links == 0:
        linked_pages = corpus.keys()
        num_links = num_pages

    #Now we have to build the probability distribution based on the links
    for p in corpus:
        prob_dist[p] = (1 - damping_factor) / num_pages
        if p in linked_pages:
            prob_dist[p] += damping_factor / num_links

    return prob_dist

    




def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #initialize the dictionary
    page_rank = {page: 0 for page in corpus}
    #initialize with a random page
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        #Count the visit to the current page
        page_rank[current_page] += 1

        #Get the transition model of the current page
        probs = transition_model(corpus, current_page, damping_factor)

        #Choose the next page based on the transition model
        current_page = random.choices(list(probs.keys()), weights=probs.values(), k=1)[0]

    #Normalize the page rank
    total_visits = sum(page_rank.values())
    page_rank = {page: visits/total_visits for page, visits in page_rank.items()}

    return page_rank

        


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_page_rank = {page: 0 for page in corpus}
    convergence_threshold = 0.001  # Adjust if necessary
    
    while True:
        for page in corpus:
            # Start with the random surfing component
            new_rank = (1 - damping_factor) / num_pages

            # Add contributions from all pages that link to `page`
            for possible_linker in corpus:
                if page in corpus[possible_linker]:
                    # If `possible_linker` links to `page`, contribute its rank divided by its number of links
                    new_rank += damping_factor * page_rank[possible_linker] / len(corpus[possible_linker])
                elif len(corpus[possible_linker]) == 0:
                    # If `possible_linker` has no links, treat it as linking to all pages
                    new_rank += damping_factor * page_rank[possible_linker] / num_pages

            new_page_rank[page] = new_rank

        # Check for convergence
        if all(abs(new_page_rank[page] - page_rank[page]) < convergence_threshold for page in corpus):
            break

        # Update the PageRank for the next iteration
        page_rank = new_page_rank.copy()

    return page_rank
    


if __name__ == "__main__":
    main()