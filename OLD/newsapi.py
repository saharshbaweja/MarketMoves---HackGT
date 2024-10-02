import requests
import re
import nltk
import numpy as np
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import LongformerTokenizer, LongformerForSequenceClassification, pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download necessary resources (only run this once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

tokenizer = T5Tokenizer.from_pretrained("t5-base")  # Use the appropriate model
model = T5ForConditionalGeneration.from_pretrained("t5-base")  # Use the appropriate model



def clean_text(text):
    # Step 1: Convert all text to lowercase
    text = text.lower()
    
    # Step 2: Tokenize text into words (tokens)
    tokens = word_tokenize(text)
    
    # Step 3: Remove stopwords (common words like "the", "and")
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Step 4: Lemmatization (reduce words to their base form)
    lemmatizer = WordNetLemmatizer()
    base_form_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return base_form_tokens

def join_tokens(tokens):
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Function to get news articles
def get_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])            

    return articles


def calculate_action_rating(sentiment_scores):
    # Unpack sentiment scores
    neg_scores = np.array([score['neg'] for score in sentiment_scores])
    neu_scores = np.array([score['neu'] for score in sentiment_scores])
    pos_scores = np.array([score['pos'] for score in sentiment_scores])
    comp_scores = np.array([score['compound'] for score in sentiment_scores])

    # Step 1: Define weights for each sentiment type
    neg_weight = -3.0  # Heavily penalize negative sentiment
    neu_weight = 0.0   # Neutral sentiment has no impact
    pos_weight = 1.0   # Positive sentiment contributes positively

    # Step 2: Calculate the weighted scores
    weighted_scores = (
        (neg_scores * neg_weight) + 
        (neu_scores * neu_weight) + 
        (pos_scores * pos_weight) +
        comp_scores
    )

    # Step 3: Calculate average score
    average_score = np.mean(weighted_scores)

    # Step 4: Set fixed thresholds for action rating
    strongbuy_threshold = 0.7
    buy_threshold = 0.5  # Threshold for Buy action
    sell_threshold = 0.3  # Threshold for Sell action
    strongsell_threshold = 0.1

    # Step 5: Determine buy/sell/neutral rating based on fixed thresholds
    if average_score > buy_threshold:
        rating = "Buy"
        if average_score > strongbuy_threshold:
            rating = "Strong Buy"
    elif average_score < sell_threshold:
        rating = "Sell"
        if average_score < strongsell_threshold:
            rating = "Strong Sell"
    else:
        rating = "Neutral"

    return average_score, rating

def analyze_sentiment_VADER(url):
    article = Article(url)

    article.download()
    article.parse()
    article.nlp()

    text = article.text
    #print(text)
    cleaned_text_array = clean_text(text)
    cleaned_text = join_tokens(cleaned_text_array)

    sentiment = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
    return(sentiment)

def generate_related_keywords(ticker):
    related_keywords = {
    "AAPL": ["Apple", "iPhone", "iPad", "MacBook", "iMac", "iOS", "App Store", "technology", "innovation", "electronics"],
    "MSFT": ["Microsoft", "Windows", "Office", "Azure", "cloud computing", "software", "Surface", "Xbox", "gaming", "productivity"],
    "GOOGL": ["Google", "search engine", "YouTube", "advertising", "Alphabet", "cloud services", "Android", "AI", "Big Data"],
    "AMZN": ["Amazon", "e-commerce", "AWS", "Kindle", "Prime", "retail", "fulfillment", "cloud computing", "logistics", "delivery"],
    "TSLA": ["Tesla", "electric vehicle", "Model S", "Model 3", "renewable energy", "autonomous driving", "Gigafactory", "batteries"],
    "FB": ["Meta", "Facebook", "Instagram", "social media", "virtual reality", "metaverse", "advertising", "WhatsApp", "connectivity"],
    "BRK.B": ["Berkshire Hathaway", "Warren Buffett", "investment", "insurance", "diversification", "holdings", "financial services"],
    "JNJ": ["Johnson & Johnson", "healthcare", "pharmaceuticals", "medical devices", "consumer products", "COVID-19", "vaccines"],
    "V": ["Visa", "payments", "credit card", "financial services", "e-commerce", "transaction processing", "digital payments"],
    "PG": ["Procter & Gamble", "consumer goods", "cleaning products", "personal care", "health", "beauty", "detergents"],
    "NVDA": ["NVIDIA", "graphics", "GPUs", "gaming", "AI", "deep learning", "data centers", "automotive", "visual computing"],
    "JPM": ["JPMorgan Chase", "banking", "financial services", "investment banking", "asset management", "loans", "credit"],
    "UNH": ["UnitedHealth Group", "healthcare", "insurance", "health plans", "medical services", "wellness", "pharmaceuticals"],
    "HD": ["Home Depot", "home improvement", "construction", "tools", "gardening", "appliances", "DIY"],
    "DIS": ["Disney", "entertainment", "theme parks", "media", "Marvel", "Star Wars", "streaming", "Disney+"]
    }   
    
    return [keyword.lower() for keyword in related_keywords.get(ticker, [])]

def is_relevant_article(article, ticker):
    keywords = [ticker] + generate_related_keywords(ticker)  # Add more as needed
    title = article.get("title", "").lower()
    description = article.get("description", "").lower()

    return any(keyword in title or keyword in description for keyword in keywords)


# def text_summary(url):
#     article = Article(url)
    
#     article.download()
#     article.parse()

#     try:
#         model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
#         tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#         # Tokenize input text
#         inputs = tokenizer.encode("summarize: " +
#                               article.text,
#                               return_tensors="pt",
#                               max_length=1024, truncation=True)
#         summary_ids = model.generate(inputs, max_length=maxSummarylength,
#                                  min_length=int(maxSummarylength/5),
#                                  length_penalty=10.0,
#                                  num_beams=4, early_stopping=True)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         print(summary)
        
#         return summary
    
#     except Exception as e:
#         print(f"Error generating summary: {e}")
#         return None
# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")  # Use the appropriate model
model = T5ForConditionalGeneration.from_pretrained("t5-base")  # Use the appropriate model

def summarize(text, maxSummarylength=500):
    # Encode the text and summarize
    inputs = tokenizer.encode("summarize: " + text,
                              return_tensors="pt",
                              max_length=1024, truncation=True)
    
    summary_ids = model.generate(inputs, max_length=maxSummarylength,
                                 min_length=int(maxSummarylength / 5),
                                 length_penalty=10.0,
                                 num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def split_text_into_pieces(text, max_tokens=900, overlapPercent=10):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Calculate the overlap in tokens
    overlap_tokens = int(max_tokens * overlapPercent / 100)

    # Split the tokens into chunks of size max_tokens with overlap
    pieces = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap_tokens)]

    # Convert the token pieces back into text
    text_pieces = [tokenizer.decode(tokenizer.convert_tokens_to_ids(piece), skip_special_tokens=True) for piece in pieces]

    return text_pieces

def text_summary(text, max_length=200, recursionLevel=0):
    recursionLevel += 1
    print("######### Recursion level:", recursionLevel, "\n\n######### ")
    
    tokens = tokenizer.tokenize(text)
    expectedCountOfChunks = len(tokens) / max_length
    max_length = int(len(tokens) / expectedCountOfChunks) + 2

    # Break the text into pieces of max_length
    pieces = split_text_into_pieces(text, max_tokens=max_length)

    print("Number of pieces:", len(pieces))
    # Summarize each piece
    summaries = []
    for k in range(len(pieces)):
        piece = pieces[k]
        # print("****************************************************")
        # print("Piece:", (k + 1), "out of", len(pieces), "pieces")
        # print(piece, "\n")
        
        summary = summarize(piece, maxSummarylength=max_length / 3 * 2)
        print("SUMMARY:", summary)
        summaries.append(summary)
        print("****************************************************")

    concatenated_summary = ' '.join(summaries)

    tokens = tokenizer.tokenize(concatenated_summary)

    if len(tokens) > max_length:
        # If the concatenated_summary is too long, repeat the process
        print("############# GOING RECURSIVE ##############")
        return recursive_summarize(concatenated_summary, max_length=max_length, recursionLevel=recursionLevel)
    else:
        # Concatenate the summaries and summarize again
        final_summary = concatenated_summary
        if len(pieces) > 1:
            final_summary = summarize(concatenated_summary, maxSummarylength=max_length)
        return final_summary


def check_input(stock_ticker):
    top_stocks = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "GOOGL",  # Alphabet Inc. (Google)
    "AMZN",  # Amazon.com Inc.
    "TSLA",  # Tesla Inc.
    "FB",    # Meta Platforms Inc. (Facebook)
    "BRK.B", # Berkshire Hathaway Inc.
    "JNJ",   # Johnson & Johnson
    "V",     # Visa Inc.
    "PG",    # Procter & Gamble Co.
    "NVDA",  # NVIDIA Corporation
    "JPM",   # JPMorgan Chase & Co.
    "UNH",   # UnitedHealth Group Incorporated
    "HD",    # Home Depot Inc.
    "DIS",   # The Walt Disney Company
    ]

    if stock_ticker not in top_stocks:
        print("Invalid stock ticker. Please enter a valid stock ticker from the top 15 US stocks.")
        return None
    return stock_ticker


def main():
    api_key = "384044306ee74b1687407339e64b02cf"

    stock_ticker = None
    while not stock_ticker:
        user_input = input("Enter the stock ticker: ").upper()
        stock_ticker = check_input(user_input)
        
    checked_ticker = check_input(stock_ticker)
    articles = get_news(stock_ticker, api_key)

    

    
    if not articles:
        print("No articles found for this stock ticker.")
        return

    urls = []
    for article in articles:
        if 'url' in article:
            urls.append(article['url'])

    count = 0
    outputs = []
    #loop through articles_urls
    largeText = ""
    for url in urls:
        try:
        # Perform sentiment analysis on the article
            if count <= 25:
                sentiment = analyze_sentiment_VADER(url)

                article = Article(url)
                article.download()
                article.parse()

                summary = text_summary(article.text)
                print(summary)
            
                # count +=1
            else: break
    
        except Exception as e:
            pass
    

    #compound_values = [output['compound'] for output in outputs]
    average_score, action = calculate_action_rating(outputs)
    average_score_rounded = round(average_score, 3)
    print(f"The overall sentiment analysis rating is {average_score_rounded}")
    print("The action you should take is: " + action + " (Not financial advice, At your own risk.)")


if __name__ == "__main__":
    main()

