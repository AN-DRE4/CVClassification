from spacy import displacy
import streamlit as st
import streamlit.components.v1 as components

def display_spacy(doc):
    # Generate HTML visualization
    html = displacy.render(doc, style="ent")
    
    # Display in Streamlit using a custom component
    components.html(html, height=400, scrolling=True)