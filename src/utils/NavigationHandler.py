import streamlit as st


def disable_navigation_if_processing():
    # st.navigation/st.Page expose no `disabled` flag, so the page switcher
    # is neutralised with CSS while inference or vectorization runs — kept
    # visible but greyed out and non-clickable, like the other disabled widgets.
    if st.session_state.get("is_processing", False):
        st.markdown(
            """
            <style>
            [data-testid="stSidebarNav"] {
                pointer-events: none;
                opacity: 0.4;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
