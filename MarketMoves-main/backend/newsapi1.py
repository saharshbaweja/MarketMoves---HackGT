import requests
import re
import nltk
import numpy as np
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import random
import yfinance as yf

# Download necessary resources (only need to run these once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    base_form_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return base_form_tokens

def join_tokens(tokens):
    return ' '.join(tokens)

# Function to get news articles for a given stock ticker
def get_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])            
    return articles

# Function to get stock data
def get_stock_data(ticker, period="1mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist['Close']

# Function to calculate indicators
def calculate_sma(prices, window):
    return prices.rolling(window=window).mean()

def calculate_ema(prices, window):
    return prices.ewm(span=window, adjust=False).mean()

# Function to calculate an action rating based on sentiment scores
def calculate_action_rating(sentiment_scores, ticker):
    neg_scores = np.array([score['neg'] for score in sentiment_scores])
    neu_scores = np.array([score['neu'] for score in sentiment_scores])
    pos_scores = np.array([score['pos'] for score in sentiment_scores])
    comp_scores = np.array([score['compound'] for score in sentiment_scores])

    neg_weight = -3.0
    neu_weight = 0.0
    pos_weight = 1.0

    weighted_scores = (
        (neg_scores * neg_weight) + 
        (neu_scores * neu_weight) + 
        (pos_scores * pos_weight) +
        comp_scores
    )

    average_score = np.mean(weighted_scores)
    
    stock_close_prices = get_stock_data(ticker)
    sma_10 = calculate_sma(stock_close_prices, window=10)
    ema_10 = calculate_ema(stock_close_prices, window=10)

    if sma_10.iloc[-1] > ema_10.iloc[-1]:
        buy_threshold = 0.6  # More aggressive buy if trend is positive
        sell_threshold = 0.3
    else:
        buy_threshold = 0.8  # More cautious buy if trend is negative
        sell_threshold = 0.5
        
    strongbuy_threshold = buy_threshold + 0.2
    strongsell_threshold = sell_threshold - 0.2

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
    cleaned_text_array = clean_text(text)
    cleaned_text = join_tokens(cleaned_text_array)
    sentiment = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
    return sentiment

def check_input(ticker):
    top_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB",
        "BRK.B", "JNJ", "V", "PG", "NVDA", "JPM", "UNH", "HD", "DIS"
    ]
    if ticker not in top_stocks:
        return None
    return ticker

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


# Define a main function that can be called from the Flask app
def analyze_stock(ticker):
    api_key = "384044306ee74b1687407339e64b02cf"

    # Validate the stock ticker
    checked_ticker = check_input(ticker)
    if not checked_ticker:
        return None, "Invalid stock ticker."

    # Retrieve news articles for the ticker
    articles = get_news(checked_ticker, api_key)
    if not articles:
        return None, "No articles found for this stock ticker."

    # Get the URLs for each article
    urls = [article['url'] for article in articles if 'url' in article]

    # Collect sentiment scores using a try-except block to skip URLs with errors
    sentiment_scores = []
    for url in urls[:25]:  # Limit to 25 articles
        try:
            sentiment = analyze_sentiment_VADER(url)
            sentiment_scores.append(sentiment)
        except Exception as e:
            print(f"Error processing sentiment for {url}: {e}")
            continue  # Skip to the next URL

    # If no valid sentiment scores are found, return a message
    if not sentiment_scores:
        return None, "No valid sentiment scores found."

    # Randomly select 5 articles for summarization
    random_urls = random.sample(urls, min(len(urls), 5))

    # Calculate the average score and determine the action to take
    average_score, action = calculate_action_rating(sentiment_scores, checked_ticker)
    print(average_score, action)
    return (round(average_score, 3), action), None
