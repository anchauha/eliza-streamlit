import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk, word_tokenize
from nltk.corpus import stopwords
import streamlit as st

# Ensure the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Ensure the necessary NLTK resources are downloaded
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Quotes for emotions
# Quotes for emotions
quotes = {
    'sad': [
        "This too shall pass.",
        "Every cloud has a silver lining.",
        "The darkest hours have only sixty minutes. — Morris Mandel",
        "Sadness flies away on the wings of time. — Jean de La Fontaine",
        "What you have to understand is that three bad quarters do not make a trend. — Kendrick Lamar",
        "Let your tears come. Let them water your soul. — Eileen Mayhew",
        "It's okay to not be okay as long as you are not giving up.",
        "Tough times never last but tough people do. — Robert H. Schuller",
        "Sometimes when you're in a dark place you think you've been buried, but you've actually been planted. — Christine Caine",
    ],
    'happy': [
        "Happiness is not by chance, but by choice. — Jim Rohn",
        "The most wasted of days is one without laughter. — E. E. Cummings",
        "Happiness radiates like the fragrance from a flower and draws all good things towards you. — Maharishi Mahesh Yogi",
        "Be happy for this moment. This moment is your life. — Omar Khayyam",
        "Happiness is when what you think, what you say, and what you do are in harmony. — Mahatma Gandhi",
    ],
    'angry': [
        "For every minute you remain angry, you give up sixty seconds of peace of mind. — Ralph Waldo Emerson",
        "Holding onto anger is like drinking poison and expecting the other person to die. — Buddha",
        "The best fighter is never angry. — Lao Tzu",
    ],
    'excited': [
        "With every adversity, there is a seed of an equivalent or greater benefit for those who are excited about life. — W. Clement Stone",
        "Do something today that your future self will thank you for. — Sean Patrick Flanery",
        "Enthusiasm moves the world. — Arthur Balfour",
    ],
    'anxious': [
        "Anxiety does not empty tomorrow of its sorrows, but only empties today of its strength. — Charles Spurgeon",
        "You don't have to control your thoughts; you just have to stop letting them control you. — Dan Millman",
        "No need to hurry. No need to sparkle. No need to be anybody but oneself. — Virginia Woolf",
    ],
    'confused': [
        "Confusion is a word we have invented for an order which is not yet understood. — Henry Miller",
        "Sometimes confusion is the pathway to clarity. — Trevor Carss",
        "The confusion we experience in this life is a whisper of the depth of our potential. — Gregg Braden",
    ],
    # Add more emotions and their quotes as needed
}

# Emotion responses, contexts, and whether a quote has been offered
emotion_contexts = {
    'sad': {
        'responses': [
            "It sounds like you're going through a tough time. Want to talk about it?",
            "Sadness can be really difficult. I'm here for you if you need to share.",
            "I'm sorry to hear you're feeling sad. Sometimes sharing can help, what's on your mind?"
        ],
        'last_context': None,
        'quote_offered': False
    },
    'happy': {
        'responses': [
            "Happiness is such a great feeling! What's been making you feel this way?",
            "That's wonderful to hear! What's been going on that's making you happy?",
            "It's great to see you in good spirits! What's the good news?"
        ],
        'last_context': None,
        'quote_offered': False
    },
    'angry': {
        'responses': [
            "Anger can be tough to handle. What's been making you feel this way?",
            "It's okay to feel angry sometimes, but it's important to handle it in a healthy way. Want to talk about it?",
            "Anger is a natural response to frustration. Do you want to discuss what's been frustrating you?"
        ],
        'last_context': None,
        'quote_offered': False
    },
    'excited': {
        'responses': [
            "Excitement can be so energizing! What's got you feeling this way?",
            "It's great to be excited about things. Tell me more about it!",
            "Your excitement is contagious! What's happening that's exciting for you?"
        ],
        'last_context': None,
        'quote_offered': False
    },
    'anxious': {
        'responses': [
            "Anxiety can be really tough. What's on your mind that's causing you to feel this way?",
            "Feeling anxious is something many of us experience. Want to talk about what's making you anxious?",
            "When you're feeling anxious, sometimes talking about it can help. What's bothering you?"
        ],
        'last_context': None,
        'quote_offered': False
    },
    'confused': {
        'responses': [
            "Confusion can be unsettling. Do you want to talk through what's on your mind?",
            "It's completely okay to feel confused. Want to discuss what's causing these feelings?",
            "Feeling confused is a sign you're facing something complex. Care to share more about it?"
        ],
        'last_context': None,
        'quote_offered': False
    },
    'neutral': {
        'responses': [
            "I see. Do you want to talk more about it?",
            "Understood. Anything else you would like to share?",
        ],
        'last_context': None,
        'quote_offered': False
    },
    # Add other emotions as needed with their own lists of responses, 'last_context' key, and 'quote_offered' key
}

conversation_tree = {}

def update_conversation_tree(tree, path, input_data):
    node = tree
    for key in path:
        if key not in node:
            node[key] = {}
        node = node[key]
    node['input'] = input_data
    node['entities'] = identify_named_entities(input_data)

# Function to retrieve the last relevant conversation piece
def get_last_conversation_piece(tree, path):
    node = tree
    for key in path:
        node = node.get(key, {})
    return node.get('input'), node.get('entities')

def process_input(user_input):
    # Look for direct emotion statements
    for emotion in quotes.keys():
        if emotion in user_input.lower():
            return emotion, 0, emotion_contexts[emotion]['responses']

    # Tokenize and remove stop words
    words = word_tokenize(user_input)
    filtered_sentence = [w for w in words if not w.lower() in stop_words]

    # Analyze sentiment
    sentiment = sia.polarity_scores(' '.join(filtered_sentence))
    # Set thresholds for sentiment (these can be adjusted)
    if sentiment['compound'] > 0.5:
        primary_emotion = 'happy'
    elif sentiment['compound'] < -0.5:
        primary_emotion = 'sad'
    else:
        primary_emotion = 'neutral'

    return primary_emotion, sentiment['compound'], emotion_contexts[primary_emotion]['responses']

def identify_named_entities(text):
    named_entities = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                named_entities.append(' '.join(c[0] for c in chunk.leaves()))
    return named_entities

def provide_quote(emotion):
    if emotion in quotes:
        return random.choice(quotes[emotion])
    else:
        return "I don't have a quote for that, but I'm here to listen."

# Main chatbot function adapted for Streamlit
def eliza_chatbot():
    st.title("Eliza Chatbot")

    # Initialize session state variables if they don't exist
    if 'name' not in st.session_state:
        st.session_state['name'] = ''
    if 'current_emotion' not in st.session_state:
        st.session_state['current_emotion'] = 'neutral'
    if 'previous_emotion' not in st.session_state:
        st.session_state['previous_emotion'] = 'neutral'
    if 'previous_sentiment_score' not in st.session_state:
        st.session_state['previous_sentiment_score'] = 0
    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = set()

    # Ask for user's name if not provided yet
    if st.session_state['name'] == '':
        st.session_state['name'] = st.text_input("What is your name?").strip()
        if st.session_state['name']:
            st.write(f"Nice to meet you, {st.session_state['name']}. How are you feeling today?")
        return

    # Main conversation loop
    user_input = st.text_input("You can share your feelings or just say anything that's on your mind.", key="user_input").strip()

    if user_input:
        # Check for conversation termination
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            st.write("Goodbye! It was nice talking to you.")
            st.session_state.clear()  # Reset all session state
            return

        # Handle repeated inputs
        if user_input.lower() in st.session_state['previous_inputs']:
            st.write("You've mentioned that before. Would you like to explore something else or go deeper into this topic?")
            return
        else:
            st.session_state['previous_inputs'].add(user_input.lower())

        # Process user input
        emotion, sentiment_score, responses = process_input(user_input)
        named_entities = identify_named_entities(user_input)

        # Update emotions and sentiment score
        if abs(sentiment_score - st.session_state['previous_sentiment_score']) > 0.4:
            st.session_state['previous_emotion'] = st.session_state['current_emotion']
            st.session_state['current_emotion'] = emotion
            st.session_state['previous_sentiment_score'] = sentiment_score

        # Offer a quote based on emotion
        # Offer a quote based on emotion
        # Offer a quote based on emotion  
        if emotion != 'neutral' and not emotion_contexts[emotion]['quote_offered']:
            quote_choice = st.radio("Would you like to hear a quote?", ('Yes', 'No'), key="quote_choice")
            if quote_choice == 'Yes':
                st.write(provide_quote(emotion))
                emotion_contexts[emotion]['quote_offered'] = True
                st.write("Do you feel a bit better or would you like to keep talking?")
            elif quote_choice == 'No':
                st.write("Let's continue our conversation.")

        # Generate response
        if st.session_state['previous_emotion'] == st.session_state['current_emotion']:
            response = random.choice(responses)
        else:
            response = "It seems like there might be a change in how you're feeling. Can you tell me more?"
        if named_entities:
            response += f" It must be significant for you to mention {', '.join(named_entities)}."
        st.write(response)
    

# Run the chatbot
if __name__ == "__main__":
    eliza_chatbot()
