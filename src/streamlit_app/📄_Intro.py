import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Belgian House Price Predictor",
    page_icon="🏠",
    initial_sidebar_state="expanded",
    layout="centered",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/Belgium-House-Price-App",
        "About": "Predict Belgian House Prices with Immoweb Data and CatBoost",
    },
)


def main():
    st.title("Belgian House Price Predictor")

    st.markdown(
        """
        [![GitHub](https://img.shields.io/github/stars/adamcseresznye/Belgium-House-Price-App?style=social)](https://github.com/adamcseresznye/Belgium-House-Price-App)
        [![Twitter](https://img.shields.io/twitter/follow/csenye22?style=social)](https://www.twitter.com/csenye22)

        Welcome to the **Belgian House Price Predictor**! This app is designed to predict house prices in Belgium using data collected from [immoweb.be](https://www.immoweb.be/en), a prominent real estate platform in the country. By employing a `CatBoost model` in conjunction with `MAPIE`, our goal is to provide precise and up-to-date price forecasts, along with an estimation of uncertainty based on *conformal prediction*.
"""
    )

    st.markdown(
        """
        Explore the current housing market, gain insights into the key determinants of property prices, and put our prediction tool to the test.
        """
    )

    st.header("Key Features")
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.markdown("🔍 **Explore Market Trends**")
    with col2:
        st.markdown("💡 **Gain Price Insights**")
    with col3:
        st.markdown("🧮 **Make Predictions**")

    st.header("Getting Started")
    st.markdown(
        """
        1. **Explore Data**: Visit the [Explore data](https://belgian-house-price-predictor.streamlit.app/Explore_data) page to understand factors influencing Belgian house prices.
        2. **Make Predictions**: Use the [Make predictions](https://belgian-house-price-predictor.streamlit.app/Make_predictions) page to estimate prices based on property features.
        3. **Have fun** 🎈🎉😊🐍💻🎈
        """
    )

    st.header("Frequently Asked Questions")
    faq = {
        "What's new in this version?": "We are excited to introduce the revamped version of our application, entirely rebuilt from scratch, featuring direct upload of sanitized data to our `MongoDB database`, removal of unnecessary features, incorporation of automatic feature selection with hyperparameter optimization using `GridSearchCV`, and predictions with 90% confidence intervals thanks to *conformal prediction*.",
        "How often is the model updated?": "Our pipeline runs monthly via GitHub actions to keep the model and dataset current.",
        "What's planned for future updates?": "We are exploring the inclusion of advertisement time to account for seasonality in the model and faster data scraping techniques. Stay tuned for updates!",
        "How does the data pipeline work?": {
            "text": "Our data processing pipeline adheres to the traditional *Extract, Transform, and Load (ETL)* process. Initially, we extract data from the source using the `requests-html` library. Following this, we execute multiple steps to refine the raw data, encompassing datatype conversions from strings to numerical values and converting low cardinal numerical data to boolean. Once our data is prepared, we perform the essential step of splitting it into train and test sets. This step ensures an unbiased model performance evaluation later on. It's worth noting that during the project's experimental phases, we evaluated various ML algorithms, including `decision trees`, `XGBoost`, and more. After rigorous experimentation, we selected `Catboost` due to its exceptional overall performance. Upon loading our pre-defined and optimized hyperparameters, we are fully equipped to train and subsequently assess the model's performance using the test set. The results of the model's performance are saved for reference. This entire pipeline is initiated on a monthly basis through GitHub actions, ensuring that both our model and dataset remain up-to-date.",
            "image": "20240804_diagram.png",
        },
        "Where can I learn more about the pipeline?": "Check out our [seven-part series of articles](https://adamcseresznye.github.io/blog/projects.html) for an in-depth look at our the data processing pipeline.",
    }

    for question, answer in faq.items():
        with st.expander(question):
            if isinstance(answer, dict):
                st.write(answer["text"])
                file_path = Path(__file__).parent
                image_path = file_path.joinpath(answer["image"])
                image = Image.open(image_path)
                st.image(
                    image,
                    caption="Data Pipeline Workflow",
                    use_column_width=True,
                )
            else:
                st.write(answer)

    st.caption(
        """
        Disclaimer: This app is for educational purposes only. The developer is not liable for any decisions made based on its outputs.
        """
    )


if __name__ == "__main__":
    main()
