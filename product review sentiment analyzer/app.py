import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Page config
st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Product Review Sentiment Analyzer")
st.markdown("### Analyze customer reviews from e-commerce platforms")

# Sidebar
with st.sidebar:
    st.header("📝 Instructions")
    st.markdown("""
    1. Paste reviews (one per line)
    2. Click 'Analyze Reviews'
    3. View sentiment breakdown
    4. Explore keyword insights
    """)
    
    st.markdown("---")
    st.markdown("**Sample Data Available**")
    if st.button("Load Sample Reviews"):
        st.session_state['load_sample'] = True

# Sample reviews
SAMPLE_REVIEWS = """Great product! Loved the quality and fast delivery.
Terrible experience. Product broke after 2 days.
It's okay, nothing special but works fine.
Amazing! Best purchase ever. Highly recommended.
Not worth the money. Very disappointed.
Good quality but delivery was slow.
Perfect! Exactly what I needed.
Average product, nothing extraordinary.
Worst purchase ever. Don't buy this.
Excellent quality and great customer service!
The fabric is soft and comfortable.
Very poor packaging, item was damaged.
Satisfied with my purchase overall.
Overpriced for what you get.
Superb! Exceeded my expectations."""

# Initialize session state
if 'load_sample' not in st.session_state:
    st.session_state['load_sample'] = False

# Text input
default_text = SAMPLE_REVIEWS if st.session_state['load_sample'] else ""
reviews_text = st.text_area(
    "Enter Reviews (one per line):",
    value=default_text,
    height=200,
    placeholder="Paste your reviews here..."
)

# Reset sample flag
if reviews_text != SAMPLE_REVIEWS:
    st.session_state['load_sample'] = False

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_sentiment(text):
    """Get sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

def extract_keywords(reviews_list, top_n=15):
    """Extract top keywords from reviews"""
    stop_words = {'the', 'is', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'a', 'an', 'it', 'was', 'this', 
                  'that', 'with', 'as', 'by', 'from', 'be', 'are'}
    
    all_words = []
    for review in reviews_list:
        words = clean_text(review).split()
        all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    return Counter(all_words).most_common(top_n)

# Analyze button
if st.button("🔍 Analyze Reviews", type="primary"):
    if not reviews_text.strip():
        st.warning("⚠️ Please enter some reviews to analyze!")
    else:
        with st.spinner("Analyzing reviews..."):
            # Process reviews
            reviews_list = [r.strip() for r in reviews_text.split('\n') if r.strip()]
            
            # Get sentiments
            results = []
            for review in reviews_list:
                sentiment, score = get_sentiment(review)
                results.append({
                    'Review': review,
                    'Sentiment': sentiment,
                    'Score': score
                })
            
            df = pd.DataFrame(results)
            
            # Display metrics
            st.markdown("---")
            st.subheader("📈 Sentiment Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            positive_count = len(df[df['Sentiment'] == 'Positive'])
            neutral_count = len(df[df['Sentiment'] == 'Neutral'])
            negative_count = len(df[df['Sentiment'] == 'Negative'])
            total_count = len(df)
            
            col1.metric("Total Reviews", total_count)
            col2.metric("✅ Positive", positive_count, 
                       f"{(positive_count/total_count*100):.1f}%")
            col3.metric("⚪ Neutral", neutral_count,
                       f"{(neutral_count/total_count*100):.1f}%")
            col4.metric("❌ Negative", negative_count,
                       f"{(negative_count/total_count*100):.1f}%")
            
            # Visualization
            st.markdown("---")
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Sentiment Distribution")
                sentiment_counts = df['Sentiment'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = {'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'}
                ax.pie(sentiment_counts.values, 
                      labels=sentiment_counts.index,
                      autopct='%1.1f%%',
                      colors=[colors[s] for s in sentiment_counts.index],
                      startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            with col_right:
                st.subheader("☁️ Review Word Cloud")
                
                # Generate word cloud
                text = ' '.join(reviews_list)
                wordcloud = WordCloud(
                    width=800,
                    height=600,
                    background_color='white',
                    colormap='viridis'
                ).generate(text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            # Keywords
            st.markdown("---")
            st.subheader("🔑 Top Keywords")
            keywords = extract_keywords(reviews_list)
            
            keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(keyword_df['Keyword'], keyword_df['Frequency'], color='#3b82f6')
            ax.set_xlabel('Frequency')
            ax.set_title('Most Common Keywords in Reviews')
            ax.invert_yaxis()
            st.pyplot(fig)
            
            # Detailed reviews
            st.markdown("---")
            st.subheader("📝 Classified Reviews")
            
            # Color code based on sentiment
            def color_sentiment(val):
                if val == 'Positive':
                    return 'background-color: #d1fae5'
                elif val == 'Negative':
                    return 'background-color: #fee2e2'
                else:
                    return 'background-color: #fef3c7'
            
            styled_df = df[['Review', 'Sentiment', 'Score']].style.applymap(
                color_sentiment, subset=['Sentiment']
            )
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download option
            st.markdown("---")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit • Sentiment Analysis with TextBlob"
    "</div>",
    unsafe_allow_html=True
)