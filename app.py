import streamlit as st
import recommender  # Import your other Python file

st.title("🍳 Breakfast Recipe Recommender")

option = st.radio("Choose Recommendation Type:", ('By Recipe Name', 'By User ID'))

if option == 'By Recipe Name':
    recipe_names = recommender.get_recipe_names()
    selected_recipe = st.selectbox("Select a Breakfast Recipe", recipe_names)
    if st.button("Recommend Similar Recipes"):
        recommendations = recommender.recommend_recipe(selected_recipe)
        st.write("Recommended Recipes:")
        for rec in recommendations:
            st.write(f"• {rec}")

elif option == 'By User ID':
    user_ids = recommender.get_user_ids()
    selected_user = st.selectbox("Select User ID", user_ids)
    if st.button("Recommend For User"):
        recommendations = recommender.recommend_for_user(selected_user)
        st.write("Recommended Recipes:")
        for rec in recommendations:
            st.write(f"• {rec}")
