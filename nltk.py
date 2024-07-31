import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, num_sentences=3):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Calculate word frequencies
    freq = FreqDist(words)
    
    # Score sentences based on word frequencies
    scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if sentence not in scores:
                    scores[sentence] = freq[word]
                else:
                    scores[sentence] += freq[word]
    
    # Get the top sentences
    summary_sentences = nlargest(num_sentences, scores, key=scores.get)
    
    # Join the top sentences
    summary = ' '.join(summary_sentences)
    
    return summary

# Example usage
blog_post = """
Artificial Intelligence (AI) is revolutionizing the way businesses operate. From automating routine tasks to providing deep insights from data, AI is becoming an indispensable tool for companies of all sizes. Small businesses, in particular, can benefit from AI-powered tools that level the playing field with larger competitors. These tools can help with customer service, marketing optimization, and predictive analytics. As AI technology continues to advance, we can expect to see even more innovative applications that will transform the business landscape.
"""

summary = summarize_text(blog_post)
print("Summary:")
print(summary)

# Generate a Twitter-style post (280 characters max)
twitter_post = summary[:277] + "..." if len(summary) > 280 else summary
print("\nTwitter Post:")
print(twitter_post)